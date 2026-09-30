"""Automated checks of the shipped labels. They prove the files are well formed and consistent with the videos and
the review log; they do not replace a person watching the footage (examples/annotations/REVIEW.md)."""
import json
from pathlib import Path

import pytest

from elder_monitor.evaluation import COVERAGE_TOLERANCE_SEC, load_annotations

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "examples" / "annotations").glob("*.json"))
LOG = (ROOT / "examples" / "annotations" / "REVIEW.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_label_file_is_valid_and_matches_its_video(path):
    labels = load_annotations(path)
    manifest = ROOT / "examples" / "outputs" / path.stem / "run_manifest.json"
    duration = json.loads(manifest.read_text(encoding="utf-8"))["duration_sec"]
    assert abs(labels["duration"] - duration) <= COVERAGE_TOLERANCE_SEC


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_provenance_and_review_status_are_recorded(path):
    provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
    assert "AI" in provenance["labels"] or "derived" in provenance["labels"]
    row = next(line for line in LOG.splitlines() if line.startswith(f"| `{path.stem}` |"))
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    if provenance["human_review"] == "pending":
        assert cells[-1] == "pending human review"
    else:
        assert cells[4] and cells[5] and cells[-1].startswith("reviewed"), "a review needs a reviewer and a date"
