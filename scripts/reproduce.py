"""Regenerate the held-out trajectories, fit on archived TRAIN, and score the sealed policies."""

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
stop_signal = None


def request_stop(signum, frame):
    # Defer cancellation until every newly started child is registered for cleanup.
    global stop_signal
    stop_signal = signum


def log(message):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{now}] {message}", flush=True)


def run_stages(*stages, stdout=None, cwd=ROOT):
    """Run independent stages together, cancelling remaining process groups on failure."""
    running = {}
    try:
        for name, command in stages:
            if stop_signal:
                raise SystemExit(128 + stop_signal)
            log(f"Starting {name}")
            proc = subprocess.Popen(
                command, cwd=cwd, stdout=stdout, start_new_session=True
            )
            running[proc] = name
        while running:
            if stop_signal:
                raise SystemExit(128 + stop_signal)
            for proc, name in list(running.items()):
                status = proc.poll()
                if status is None:
                    continue
                if status:
                    log(f"{name} failed (exit {status})")
                    raise SystemExit(status if status > 0 else 128 - status)
                log(f"Finished {name}")
                del running[proc]
            if running:
                time.sleep(0.2)
    finally:
        # A failed or interrupted process can leave native CPU pool workers alive.
        for proc in running:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        for proc in running:
            proc.wait()


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("worker count must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(
        prog="bash scripts/reproduce.sh",
        description=__doc__,
        epilog="Set TRAIN_PYTHON to use a separate Python executable for fitting.",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path("runs/full"),
        help="New output directory (default: runs/full)",
    )
    parser.add_argument(
        "workers",
        nargs="?",
        type=positive,
        default=2,
        help="Native CPU simulation workers (default: 2)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Policy fitting device (default: cpu). CUDA overlaps held-out simulation.",
    )
    parser.add_argument(
        "--train-workers",
        type=positive,
        help="Fitting workers (CPU defaults to the simulation worker count, CUDA uses one)",
    )
    args = parser.parse_args()
    output = args.destination.absolute()
    if output.exists():
        parser.error(f"Choose a new output directory: {output}")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    python = sys.executable
    train_python = shutil.which(os.environ.get("TRAIN_PYTHON") or python)
    if train_python is None:
        parser.error("Could not find the interpreter specified by TRAIN_PYTHON")
    # Keep virtualenv symlinks and resolve relative executable paths before changing cwd.
    train_python = os.path.abspath(train_python)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    device = args.device
    if device != "cpu":
        probe = [
            train_python,
            "-c",
            "import sys\nfrom agent import resolve_device\n"
            "try: print(resolve_device(sys.argv[1]))\n"
            "except ValueError as error: sys.exit(str(error))",
            device,
        ]
        with tempfile.TemporaryFile(mode="w+") as result:
            run_stages(
                ("training device check", probe), stdout=result, cwd=ROOT / "training"
            )
            result.seek(0)
            device = result.read().strip()
        if device not in ("cpu", "cuda"):
            parser.error(
                f"Unrecognized device returned by the training interpreter: {device!r}"
            )

    output.mkdir(parents=True)
    native = str(args.workers)
    fit_workers = str(args.train_workers or (1 if device == "cuda" else args.workers))

    run_stages(
        (
            "alignment analysis",
            [python, "analysis/reproduce.py", "--output", output / "alignment.json"],
        )
    )
    for name, command, report in (
        (
            "archived contact intervention",
            [python, "mechanism/verify.py"],
            "contact-cut.json",
        ),
        (
            "native contact intervention",
            [python, "-m", "mechanism.replay", "--workers", native],
            "native-contact-cut.json",
        ),
    ):
        path = output / report
        with path.open("x") as stream:
            run_stages((name, command), stdout=stream)
        print(path.read_text(), end="", flush=True)

    grid = (
        "native held-out simulation",
        [
            python,
            "simulation/replay.py",
            "test-grid",
            "--workers",
            native,
            "--output",
            output / "native-grid.npz",
            "--report",
            output / "native-grid.json",
        ],
    )
    fit = (
        f"policy fit on {device}",
        [
            train_python,
            "training/run.py",
            "fit",
            "--out",
            output / "contact",
            "--workers",
            fit_workers,
            "--controls",
            "--device",
            device,
        ],
    )
    if device == "cuda":
        run_stages(grid, fit)
    else:
        run_stages(grid)
        run_stages(fit)
    run_stages(
        (
            "held-out evaluation",
            [
                python,
                "training/run.py",
                "evaluate",
                "--run",
                output / "contact",
                "--native-test",
                output / "native-grid.npz",
                "--json",
                output / "native-test.json",
            ],
        )
    )
    print(f"Results: {output / 'native-test.json'}")


if __name__ == "__main__":
    sys.exit(main())
