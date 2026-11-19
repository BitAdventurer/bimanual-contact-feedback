"""Plot the first archived contact-intervention case after the policy's own first action."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .verify import references


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("left", "hold", "right")


def plot(output):
    data = ROOT / "mechanism" / "data"
    case_file = sorted((data / "cases").glob("*.npz"))[0]
    reference = references(data / "reference.csv")
    family = case_file.stem
    first, = (action for action in range(3) if reference[family, 0, action, 0][0])
    with np.load(case_file, allow_pickle=False) as case:
        # The archive packs large left, stationary, large right on the motion axis.
        normal = case["force_trace"][:, first, :, :, 2]
        maps = case["p"][:, first]
    difference_mn = np.abs(maps[:, 2] - maps[:, 0]).sum(axis=(-2, -1)) * 1000
    # mechanism.replay.partner_interval records each completed MuJoCo step
    # and acquires one socket observation at the rounded end of each third.
    n_steps = normal.shape[-1]
    elapsed_ms = np.arange(1, n_steps + 1) * 0.5
    packet_steps = [round(i * n_steps / 3) for i in (1, 2, 3)]
    packet_ms = elapsed_ms[np.array(packet_steps) - 1]

    ink, blue, red = "#202F3B", "#1E6481", "#A64324"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": ink, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink}):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 4.5), sharex=True, sharey="row",
                                 gridspec_kw={"height_ratios": [1.4, 1]})
        fig.subplots_adjust(left=.17, right=.975, top=.79, bottom=.25, hspace=.23, wspace=.14)
        for world, title in enumerate(("Contact intact", "Contact disabled")):
            force_ax, map_ax = axes[:, world]
            force_ax.set_title(title, fontweight="bold", fontsize=12, pad=9, loc="left")
            right, = force_ax.plot(elapsed_ms, normal[world, 2], color=red, linewidth=2.3)
            left, = force_ax.plot(elapsed_ms, normal[world, 0], color=blue, linewidth=1.7,
                                  linestyle=(0, (4, 4)))
            map_ax.scatter(packet_ms, difference_mn[world], s=39, color=ink, zorder=3)
            choices = [ACTIONS[reference[family, world, first, motion][1]] for motion in (0, 4)]
            map_ax.text(0, -.58, "Pin left → " + choices[0] + "     Pin right → " + choices[1],
                        transform=map_ax.transAxes, fontsize=10, fontweight="semibold")

        axes[0, 0].set_ylabel("Pin–socket\nnormal force (N)")
        axes[1, 0].set_ylabel("Socket pad-map\ndifference (mN)")
        for axis in axes.flat:
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.spines["bottom"].set_color("#B5C3C9")
            axis.grid(axis="y", color="#DBE2E5", linewidth=.75)
            axis.set_axisbelow(True)
            axis.tick_params(length=0, pad=4)
            axis.set_xlim(-3, 330)
        axes[0, 0].set(ylim=(-.15, 3.15), yticks=[0, 1, 2, 3])
        axes[1, 0].set(ylim=(-.34, 5), yticks=[0, 2, 4], xticks=[0, 80, 160, 240, 320])
        for axis in axes[1]:
            axis.set_xlabel("Time after pin command (ms)", labelpad=4)
        fig.legend([left, right], ["Pin commanded left", "Pin commanded right"], ncol=2,
                   loc="upper center", bbox_to_anchor=(.56, .996), frameon=False,
                   columnspacing=2.7, handlelength=3)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.13)
        plt.close(fig)
    print(f"{family}: first action {ACTIONS[first]}. Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/media/contact-traces.png")
    plot(parser.parse_args().output)
