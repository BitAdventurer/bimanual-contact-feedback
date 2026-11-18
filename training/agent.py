"""Two-step on-policy learner. The learner can only query the chosen transition."""

from pathlib import Path
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


class CachedContact:
    def __init__(self, root, mid, success, rng):
        self.root = root
        self._mid = mid
        self._success = success
        self._rng = rng

    def reset(self, n):
        self._family = self._rng.integers(len(self.root), size=n)
        self._future = self._rng.integers(5, size=n)
        return self.root[self._family]

    def first(self, action, reveal=True):
        self._a = action
        if reveal:
            return self._mid[self._family, self._future, action]

    def finish(self, action):
        y = self._success[self._family, self._future, self._a, action]
        log = np.stack((self._family, self._future, self._a, action, y), axis=1).astype(
            np.int16
        )
        return torch.from_numpy(y.astype(np.float32)), log


class Policy(nn.Module):
    def __init__(self, n, options):
        super().__init__()
        self.hidden1 = nn.Linear(n, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.actor = nn.Linear(32, options)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        h = torch.tanh(self.hidden2(torch.tanh(self.hidden1(x))))
        return self.actor(h), self.critic(h)[:, 0]

    def export(self, prefix):
        out = {}
        for i, layer in enumerate((self.hidden1, self.hidden2, self.actor)):
            out[prefix + "W" + str(i)] = layer.weight.detach().numpy()
            out[prefix + "b" + str(i)] = layer.bias.detach().numpy()
        return out


def act(net, x):
    logits, v = net(x)
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return a, dist.log_prob(a), v, dist.entropy().mean().item()


def ppo_loss(net, data, idx, cfg):
    x, a, old_log, adv, ret = data
    logits, v = net(x[idx])
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(a[idx])
    ratio = (logp - old_log[idx]).exp()
    filtered = ratio.clamp(1 - cfg["ppo_clip_ratio"], 1 + cfg["ppo_clip_ratio"])
    actor = -torch.minimum(ratio * adv[idx], filtered * adv[idx]).mean()
    return (
        actor
        + cfg["value_coefficient"] * (v - ret[idx]).square().mean()
        - cfg["entropy_coefficient"] * dist.entropy().mean()
    )


def advantage(x, a, logp, v, reward):
    adv = reward - v
    return (
        x,
        a,
        logp,
        (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1.0e-8),
        reward,
    )


def fit(inp):
    method, mode, seed, training, roots, mids, success, dest = inp
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    root = torch.from_numpy(roots)
    mid = torch.from_numpy(mids)
    env = CachedContact(root, mid, success, rng)
    committed = method.endswith("_committed")
    sampled = method.startswith("ppo")
    net = Policy(root.shape[-1], 9 if committed else 3)
    feedback = Policy(mid.shape[-1], 3) if not committed else None
    all_params = list(net.parameters()) + (
        [] if committed else list(feedback.parameters())
    )
    opt = torch.optim.Adam(all_params, lr=training["learning_rate"])
    batch = training["batch_episodes"]
    mb = training["minibatch"]
    if not sampled:
        teachers = torch.from_numpy(success.astype(np.float32))
        target_root = (
            teachers.mean(1).reshape(len(root), 9)
            if committed
            else teachers.max(-1).values.mean(1)
        )
    else:
        teachers = None
    metadata = []
    traj = []
    folder = Path(dest) / mode / method / str(seed)
    folder.mkdir(parents=True)
    for update in range(1, training["updates"] + 1):
        if sampled:
            x = env.reset(batch)
            with torch.no_grad():
                action, logp, v, ent = act(net, x)
                a = action.numpy() // 3 if committed else action.numpy()
                m = env.first(a, reveal=not committed)
                if committed:
                    b = action.numpy() % 3
                    ent2 = 0.0
                else:
                    b_t, lp_m, v_m, ent2 = act(feedback, m)
                    b = b_t.numpy()
                reward, replay = env.finish(b)
                traj.append(replay)
                stage1 = advantage(x, action, logp, v, reward)
                if not committed:
                    stage2 = advantage(m, b_t, lp_m, v_m, reward)
            record = dict(
                update=update,
                sampled_reward=float(reward.mean()),
                root_entropy=ent,
                mid_entropy=ent2,
            )
        else:
            i = rng.integers(len(root), size=batch)
            x = root[i]
            root_y = target_root[i]
            if not committed:
                z = rng.integers(5, size=batch)
                a = rng.integers(3, size=batch)
                m = mid[i, z, a]
                mid_y = teachers[i, z, a]
            record = dict(update=update)
        for _ in range(training["epochs"]):
            order = rng.permutation(batch)
            for off in range(0, batch, mb):
                idx = order[off : off + mb]
                if sampled:
                    loss = ppo_loss(net, stage1, idx, training)
                    if not committed:
                        loss = loss + ppo_loss(feedback, stage2, idx, training)
                else:
                    loss = (net(x[idx])[0] - root_y[idx]).square().mean()
                    if not committed:
                        loss = loss + (feedback(m[idx])[0] - mid_y[idx]).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, training["gradient_norm"])
                opt.step()
        record["last_minibatch_loss"] = float(loss.detach())
        metadata.append(record)
        if update in training["checkpoints"]:
            weights = net.export("r")
            if not committed:
                weights.update(feedback.export("m"))
            with (folder / ("step_%03d.npz" % update)).open("xb") as f:
                np.savez_compressed(f, **weights)
    if sampled:
        with (folder / "onpolicy.npz").open("xb") as f:
            np.savez_compressed(f, transitions=np.stack(traj))
    return mode, method, seed, metadata
