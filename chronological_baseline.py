"""Train and evaluate a fixed chronological FI-2010 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COUNT = 144
ROW_COUNT = 149
CLASSES = (1, 2, 3)
STATIONARY = 2
SUBSET = "Dst_NoAuction_DecPre"
SPLIT_FILES = {
    "train": (f"Train_{SUBSET}_CF_7.txt",),
    "validation": (f"Test_{SUBSET}_CF_7.txt",),
    "test": (f"Test_{SUBSET}_CF_8.txt", f"Test_{SUBSET}_CF_9.txt"),
}


def read_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read 144 feature rows and the first of five label rows."""
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if data.shape[0] != ROW_COUNT or data.shape[1] == 0:
        raise ValueError(f"{path.name}: expected 149 rows and at least one observation")
    features, labels = data[:FEATURE_COUNT].T, data[FEATURE_COUNT]
    if not np.isfinite(features).all() or not np.isin(labels, CLASSES).all():
        raise ValueError(f"{path.name}: expected finite features and labels in {CLASSES}")
    return features, labels.astype(np.int64)


def resolve_paths(data_dir: Path) -> dict[str, tuple[Path, ...]]:
    paths = {}
    for split, names in SPLIT_FILES.items():
        found = []
        for name in names:
            matches = sorted(path for path in data_dir.rglob(name) if path.is_file())
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} under {data_dir}, found {len(matches)}")
            found.append(matches[0])
        paths[split] = tuple(found)
    return paths


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=CLASSES).tolist(),
    }


def run(data_dir: Path) -> dict:
    paths = resolve_paths(data_dir)
    inputs = {
        split: [{"file": p.name, "sha256": file_sha256(p)} for p in selected]
        for split, selected in paths.items()
    }
    hashes = [item["sha256"] for records in inputs.values() for item in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Two selected inputs are byte-identical; check the fold files")

    def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
        blocks = [read_file(path) for path in paths[split]]
        return np.concatenate([x for x, _ in blocks]), np.concatenate([y for _, y in blocks])

    x_train, y_train = load_split("train")
    if set(y_train) != set(CLASSES):
        raise ValueError("The training file must contain all three target classes")
    model = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    model.fit(x_train, y_train)

    reports = {}
    for split in ("validation", "test"):
        features, labels = load_split(split)
        reports[split] = {
            "observations": int(labels.size),
            "logistic_regression": evaluate(labels, model.predict(features)),
            "always_stationary": evaluate(labels, np.full_like(labels, STATIONARY)),
        }
    return {
        "target": "first FI-2010 label row (10 events)",
        "class_order": list(CLASSES),
        "training_observations": int(y_train.size),
        "inputs": inputs,
        **reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    args = parser.parse_args()
    try:
        report = run(args.data_dir)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
