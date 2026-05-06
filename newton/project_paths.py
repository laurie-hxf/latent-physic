from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEMO_PATH = REPO_ROOT / "20260406_183206.h5"
DEFAULT_DEMO_METADATA_PATH = REPO_ROOT / "20260406_183206.json"
DEFAULT_PLY_PATH = REPO_ROOT / "PushT183206" / "pointcloud_step_0560.ply"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_SCENE_USD_PATH = DEFAULT_OUTPUT_DIR / "pusht_rigid.usda"
