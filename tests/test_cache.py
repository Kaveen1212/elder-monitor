"""Observation-cache reuse is tied to the weights, perception settings and code (review of commit 4a1a280)."""
import json

import pytest

from elder_monitor.pipeline import analyze, perception_key
from elder_monitor.reporting import read_observations, write_observations
from elder_monitor.schemas import Observation
from elder_monitor.video import file_hash

from test_units import write_video


@pytest.fixture
def weights(cfg, tmp_path):
    path = tmp_path / "pose.pt"
    path.write_bytes(b"weights v1")
    cfg["vision"]["model"] = str(path)
    return path


def test_changed_weights_or_perception_settings_change_the_key(cfg, weights):
    key = perception_key(cfg)
    weights.write_bytes(b"weights v2, same file name")
    assert perception_key(cfg) != key
    weights.write_bytes(b"weights v1")
    assert perception_key(cfg) == key
    cfg["identity"]["min_det_conf"] = 0.6
    assert perception_key(cfg) != key


def test_analysis_settings_keep_the_key(cfg, weights):
    key = perception_key(cfg)
    cfg["events"]["exit_persist_sec"] = 3.0
    cfg["policy"]["max_out_of_bed_sec"] = 120
    cfg["agent"]["enabled"] = False
    assert perception_key(cfg) == key


def test_missing_weights_and_old_caches_are_never_trusted(cfg, weights, tmp_path):
    cache = tmp_path / "observations.jsonl"
    write_observations(cache, {"video_hash": "v"}, [Observation(0.0)])
    assert read_observations(cache, {"video_hash": "v", "perception_hash": perception_key(cfg)}) == (None, None)
    weights.unlink()
    assert perception_key(cfg) is None


def test_reused_observations_name_the_run_that_made_them(cfg, weights, tmp_path):
    reader = write_video(tmp_path / "clip.avi", 30, 10)
    video = tmp_path / "clip.avi"
    meta = {"video_hash": file_hash(video), "perception_hash": perception_key({**cfg, "sampling": {"fps": 5}}),
            "duration": reader.duration, "width": 64, "height": 48, "pose_model": "pose.pt sha256:abc",
            "perception_sec": 1.5, "created": "2026-01-02T03:04:05+00:00",
            "source": {"commit": "abc123", "dirty": False, "src_dirty": False}, "versions": {"python": "3.x"}}
    out = tmp_path / "out"
    out.mkdir()
    write_observations(out / "observations.jsonl", meta, [Observation(round(0.2 * k, 4), reason="not_detected")
                                                          for k in range(15)])
    analyze(video, cfg, out, use_vlm=False, reuse=True)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reused_observations"] and manifest["pose_model"] == "pose.pt sha256:abc"
    assert manifest["perception"]["created"] == meta["created"] and manifest["perception"]["source"]["commit"] == "abc123"
    assert "commit" in (manifest["source"] or {"commit": None})
