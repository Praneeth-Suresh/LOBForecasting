"""Run the baseline on synthetic files shaped like FI-2010, not market data."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from chronological_baseline import SPLIT_FILES, resolve_paths


ROOT = Path(__file__).resolve().parents[1]


def write_example(path: Path, n: int, offset: int) -> None:
    labels = np.tile([1, 2, 3], n // 3)
    data = np.zeros((149, n))
    data[0] = 20 * labels + offset
    data[1] = offset
    data[144:] = labels
    np.savetxt(path, data)


class BaselineTests(unittest.TestCase):
    def test_cli_report_is_repeatable_and_uses_all_reserved_file_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            tasks = [name for files in SPLIT_FILES.values() for name in files]
            for offset, (name, count) in enumerate(zip(tasks, (30, 12, 15, 18))):
                write_example(directory / name, count, offset)

            command = [
                sys.executable, str(ROOT / "chronological_baseline.py"),
                "--data-dir", tmp,
            ]
            result = subprocess.check_output(command, text=True)
            self.assertEqual(result, subprocess.check_output(command, text=True))
            report = json.loads(result)
            self.assertEqual(report["training_observations"], 30)
            self.assertEqual(report["validation"]["observations"], 12)
            self.assertEqual(report["test"]["observations"], 33)
            self.assertEqual(report["test"]["always_stationary"]["accuracy"], 1 / 3)
            self.assertEqual(len(report["inputs"]["test"]), 2)

    def test_ambiguous_file_is_rejected_instead_of_mixing_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "copy").mkdir()
            for folder in (directory, directory / "copy"):
                (folder / SPLIT_FILES["train"][0]).touch()
            with self.assertRaisesRegex(ValueError, "found 2"):
                resolve_paths(directory)


if __name__ == "__main__":
    unittest.main()
