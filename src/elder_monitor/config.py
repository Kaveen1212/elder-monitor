import copy
import hashlib
import json
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "default.yaml"


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None, overrides=None):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg = deep_merge(cfg, overrides)
    tracker = Path(cfg["vision"]["tracker"])
    if not tracker.exists() and (PACKAGE_DIR / tracker).exists():
        cfg["vision"]["tracker"] = str(PACKAGE_DIR / tracker)
    return cfg


def validate_scene(cfg):
    bed = cfg["scene"].get("bed_polygon")
    if not bed or len(bed) < 3:
        raise ValueError("scene.bed_polygon needs at least 3 normalised points; run `calibrate` first")
    for x, y in bed:
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("scene.bed_polygon must use normalised coordinates in [0, 1]")


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
