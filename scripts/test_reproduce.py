"""Exercise stage ordering and cleanup without training hardware or simulation data."""

import os
from pathlib import Path
import select
import sys
import tempfile
import unittest

from scripts.reproduce import run_stages


WAIT_FOR_FILE = """
import sys, time
from pathlib import Path
def wait_for(path):
    deadline = time.monotonic() + 5
    while not Path(path).exists():
        if time.monotonic() > deadline:
            sys.exit(9)
        time.sleep(0.01)
"""


class StageTests(unittest.TestCase):
    def test_independent_stages_overlap_and_both_finish(self):
        script = (
            WAIT_FOR_FILE
            + """
Path(sys.argv[1]).touch()
wait_for(sys.argv[2])
time.sleep(float(sys.argv[3]))
Path(sys.argv[1] + ".done").touch()
"""
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_stages(
                ("grid", [sys.executable, "-c", script, "grid", "fit", "0.4"]),
                ("fit", [sys.executable, "-c", script, "fit", "grid", "0"]),
                cwd=tmp,
            )
            for stage in ("grid", "fit"):
                self.assertTrue((Path(tmp) / (stage + ".done")).is_file())

    def test_failure_stops_peer_and_its_worker(self):
        launcher = """
import subprocess, sys
from pathlib import Path
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
Path("worker-started").touch()
worker.wait()
"""
        failure = WAIT_FOR_FILE + '\nwait_for("worker-started")\nsys.exit(23)'
        read_fd, write_fd = os.pipe()
        with tempfile.TemporaryDirectory() as tmp, os.fdopen(read_fd, "rb") as reader:
            with os.fdopen(write_fd, "wb") as writer, self.assertRaises(
                SystemExit
            ) as error:
                run_stages(
                    ("grid", [sys.executable, "-c", launcher]),
                    ("fit", [sys.executable, "-c", failure]),
                    cwd=tmp,
                    stdout=writer,
                )
            self.assertEqual(error.exception.code, 23)
            # The worker inherits stdout. EOF proves it also exited on peer failure.
            self.assertTrue(select.select([reader], [], [], 2)[0])
            self.assertEqual(reader.read(1), b"")


if __name__ == "__main__":
    unittest.main()
