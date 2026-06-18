from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NEWTON_DIR = REPO_ROOT / "newton"
if str(NEWTON_DIR) not in sys.path:
    sys.path.insert(0, str(NEWTON_DIR))

from mujoco_contact_friction_fit_utils import (  # noqa: E402
    MujocoTrajectory,
    MujocoTrajectoryCollection,
    load_mujoco_trajectories,
    sample_mujoco_trajectory_time_window,
    slice_mujoco_trajectory_time_window,
)


MANIFEST_SCHEMA_VERSION = 1
VALID_OBJECT_SPLITS = ("train", "validation", "test")
VALID_EPISODE_POOLS = ("context", "query", "eval")
ENCODER_FEATURE_SCHEMA = (
    "relative_position_local_x",
    "relative_position_local_y",
    "relative_yaw_sin",
    "relative_yaw_cos",
    "linear_velocity_initial_local_x",
    "linear_velocity_initial_local_y",
    "angular_velocity_z",
    "force_initial_local_x",
    "force_initial_local_y",
    "force_point_offset_local_x",
    "force_point_offset_local_y",
    "force_magnitude",
)


@dataclass(frozen=True)
class DatasetHeader:
    path: Path
    num_trajectories: int
    min_steps: int
    max_steps: int
    columns: tuple[str, ...]
    summary_metadata: dict[str, Any]


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    physical_config_id: str
    shape_id: str
    trajectory_npz: Path
    object_split: str
    context_episode_indices: tuple[int, ...]
    query_episode_indices: tuple[int, ...]
    eval_episode_indices: tuple[int, ...]
    friction_spec: dict[str, Any]
    dino_feature_npz: Path | None
    num_trajectories: int
    min_steps: int
    max_steps: int

    def pool_indices(self, pool: str) -> tuple[int, ...]:
        if pool == "context":
            return self.context_episode_indices
        if pool == "query":
            return self.query_episode_indices
        if pool == "eval":
            return self.eval_episode_indices
        raise ValueError(f"Unknown episode pool {pool!r}; expected one of {VALID_EPISODE_POOLS}")


@dataclass(frozen=True)
class EncoderFeatureBatch:
    features: np.ndarray
    valid_mask: np.ndarray
    lengths: np.ndarray
    episode_indices: np.ndarray
    window_start_steps: np.ndarray
    feature_schema: tuple[str, ...] = ENCODER_FEATURE_SCHEMA


@dataclass(frozen=True)
class QueryTrajectoryBatch:
    trajectories: tuple[MujocoTrajectory, ...]
    episode_indices: np.ndarray
    window_start_steps: np.ndarray


@dataclass(frozen=True)
class ObjectTrainingSample:
    object_spec: ObjectSpec
    context_a: EncoderFeatureBatch
    context_b: EncoderFeatureBatch
    query_a: QueryTrajectoryBatch
    query_b: QueryTrajectoryBatch


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_hash(value: Any, length: int = 12) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[: int(length)]


def _sanitize_id(value: str) -> str:
    result = []
    previous_separator = False
    for char in str(value).strip().lower():
        if char.isalnum():
            result.append(char)
            previous_separator = False
        elif not previous_separator:
            result.append("_")
            previous_separator = True
    return "".join(result).strip("_") or "object"


def _json_from_npz_scalar(data: np.lib.npyio.NpzFile, key: str, default: Any) -> Any:
    if key not in data.files:
        return default
    value = data[key].item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def inspect_dataset_header(path: Path, *, min_episode_steps: int = 1) -> DatasetHeader:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Trajectory dataset does not exist: {dataset_path}")

    with np.load(dataset_path, allow_pickle=True) as data:
        required = {"trajectories", "columns", "episode_lengths"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{dataset_path} is missing required dataset keys: {missing}")
        trajectories_shape = tuple(int(value) for value in data["trajectories"].shape)
        columns = tuple(str(value) for value in data["columns"].tolist())
        episode_lengths = np.asarray(data["episode_lengths"], dtype=np.int32).reshape(-1)
        summary_metadata = _json_from_npz_scalar(data, "summary_metadata_json", {})
        has_point_offsets_local = "point_offset_local" in data.files

    if len(trajectories_shape) != 3:
        raise ValueError(f"{dataset_path} trajectories must have rank 3, got {trajectories_shape}")
    if trajectories_shape[0] != len(episode_lengths):
        raise ValueError(
            f"{dataset_path} trajectory/episode-length mismatch: "
            f"{trajectories_shape[0]} trajectories vs {len(episode_lengths)} lengths"
        )
    if trajectories_shape[2] != len(columns):
        raise ValueError(
            f"{dataset_path} trajectory/column mismatch: width={trajectories_shape[2]} columns={len(columns)}"
        )
    if not has_point_offsets_local:
        raise ValueError(f"{dataset_path} is missing point_offset_local")

    steps = episode_lengths.astype(np.int64) - 1
    eligible = steps >= int(min_episode_steps)
    if not np.any(eligible):
        raise ValueError(f"{dataset_path} contains no episode with at least {int(min_episode_steps)} steps")
    eligible_steps = steps[eligible]
    return DatasetHeader(
        path=dataset_path,
        num_trajectories=int(len(episode_lengths)),
        min_steps=int(np.min(eligible_steps)),
        max_steps=int(np.max(eligible_steps)),
        columns=columns,
        summary_metadata=dict(summary_metadata) if isinstance(summary_metadata, dict) else {},
    )


def _dataset_eligible_indices(path: Path, *, min_episode_steps: int) -> np.ndarray:
    with np.load(Path(path), allow_pickle=True) as data:
        episode_lengths = np.asarray(data["episode_lengths"], dtype=np.int32).reshape(-1)
    return np.flatnonzero(episode_lengths - 1 >= int(min_episode_steps)).astype(np.int32)


def _infer_friction_spec(summary_metadata: dict[str, Any]) -> dict[str, Any]:
    block_friction = summary_metadata.get("block_friction_override") or summary_metadata.get("block_friction") or {}
    spec: dict[str, Any] = {
        "partition_family": summary_metadata.get("friction_partition_family"),
        "block_friction": block_friction,
    }
    if isinstance(block_friction, dict):
        left = block_friction.get("push_block_left")
        right = block_friction.get("push_block_right")
        if isinstance(left, (list, tuple)) and left:
            spec["left_mu"] = float(left[0])
        if isinstance(right, (list, tuple)) and right:
            spec["right_mu"] = float(right[0])
        if "left_mu" in spec and "right_mu" in spec:
            spec["mean_mu"] = 0.5 * (spec["left_mu"] + spec["right_mu"])
            spec["right_minus_left"] = spec["right_mu"] - spec["left_mu"]
    return spec


def _infer_shape_spec(summary_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "bounds_min": summary_metadata.get("block_local_bounds_min"),
        "bounds_max": summary_metadata.get("block_local_bounds_max"),
    }


def _allocate_counts(total: int, fractions: Sequence[float], *, require_nonempty: bool) -> list[int]:
    if total <= 0:
        raise ValueError("Cannot split an empty collection")
    values = np.asarray(fractions, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0.0) or float(np.sum(values)) <= 0.0:
        raise ValueError(f"Invalid split fractions: {fractions}")
    values = values / np.sum(values)
    raw = values * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total - int(np.sum(counts)))
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[int(index)] += 1

    positive = np.flatnonzero(values > 0.0)
    if require_nonempty and total >= len(positive):
        for index in positive:
            if counts[index] > 0:
                continue
            donors = np.argsort(-counts, kind="stable")
            donor = next((int(item) for item in donors if counts[int(item)] > 1), None)
            if donor is None:
                break
            counts[donor] -= 1
            counts[int(index)] += 1
    return [int(value) for value in counts]


def _split_episode_indices(
    eligible_indices: np.ndarray,
    *,
    fractions: Sequence[float],
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(eligible_indices, dtype=np.int32)[rng.permutation(len(eligible_indices))]
    counts = _allocate_counts(len(shuffled), fractions, require_nonempty=True)
    context_end = counts[0]
    query_end = context_end + counts[1]
    return (
        tuple(sorted(int(value) for value in shuffled[:context_end])),
        tuple(sorted(int(value) for value in shuffled[context_end:query_end])),
        tuple(sorted(int(value) for value in shuffled[query_end:])),
    )


def _assign_object_splits(
    physical_config_ids: Sequence[str],
    *,
    fractions: Sequence[float],
    seed: int,
) -> dict[str, str]:
    unique_configs = sorted(set(str(value) for value in physical_config_ids))
    rng = np.random.default_rng(int(seed))
    shuffled = [unique_configs[int(index)] for index in rng.permutation(len(unique_configs))]
    counts = _allocate_counts(len(shuffled), fractions, require_nonempty=True)
    result: dict[str, str] = {}
    cursor = 0
    for split, count in zip(VALID_OBJECT_SPLITS, counts, strict=True):
        for config_id in shuffled[cursor : cursor + count]:
            result[config_id] = split
        cursor += count
    return result


def _relative_path(path: Path, parent: Path) -> str:
    return os.path.relpath(Path(path).resolve(), start=Path(parent).resolve())


def build_manifest(
    dataset_paths: Sequence[Path],
    *,
    output_path: Path,
    seed: int = 0,
    episode_split_fractions: Sequence[float] = (0.15, 0.75, 0.10),
    object_split_fractions: Sequence[float] = (0.70, 0.15, 0.15),
    min_episode_steps: int = 1,
    dino_feature_npz: Path | None = None,
) -> dict[str, Any]:
    paths = sorted({Path(path).expanduser().resolve() for path in dataset_paths})
    if not paths:
        raise ValueError("At least one trajectory dataset is required")

    output = Path(output_path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    seen_object_ids: set[str] = set()
    for dataset_index, path in enumerate(paths):
        header = inspect_dataset_header(path, min_episode_steps=int(min_episode_steps))
        friction_spec = _infer_friction_spec(header.summary_metadata)
        shape_spec = _infer_shape_spec(header.summary_metadata)
        shape_id = f"shape_{_stable_hash(shape_spec)}"
        physical_config_id = f"physics_{_stable_hash({'shape': shape_spec, 'friction': friction_spec})}"
        base_object_id = _sanitize_id(path.stem)
        object_id = base_object_id
        if object_id in seen_object_ids:
            object_id = f"{base_object_id}_{_stable_hash(str(path), length=8)}"
        seen_object_ids.add(object_id)

        eligible_indices = _dataset_eligible_indices(path, min_episode_steps=int(min_episode_steps))
        episode_seed = int(seed) + int(_stable_hash(object_id, length=8), 16)
        context_indices, query_indices, eval_indices = _split_episode_indices(
            eligible_indices,
            fractions=episode_split_fractions,
            seed=episode_seed,
        )
        records.append(
            {
                "object_id": object_id,
                "physical_config_id": physical_config_id,
                "shape_id": shape_id,
                "trajectory_npz": _relative_path(path, output.parent),
                "dino_feature_npz": (
                    None if dino_feature_npz is None else _relative_path(Path(dino_feature_npz), output.parent)
                ),
                "object_split": None,
                "context_episode_indices": list(context_indices),
                "query_episode_indices": list(query_indices),
                "eval_episode_indices": list(eval_indices),
                "friction_spec": friction_spec,
                "num_trajectories": header.num_trajectories,
                "eligible_trajectories": int(len(eligible_indices)),
                "min_steps": header.min_steps,
                "max_steps": header.max_steps,
                "dataset_index": int(dataset_index),
            }
        )

    object_splits = _assign_object_splits(
        [str(record["physical_config_id"]) for record in records],
        fractions=object_split_fractions,
        seed=int(seed),
    )
    for record in records:
        record["object_split"] = object_splits[str(record["physical_config_id"])]

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "episode_split_fractions": {
            "context": float(episode_split_fractions[0]),
            "query": float(episode_split_fractions[1]),
            "eval": float(episode_split_fractions[2]),
        },
        "object_split_fractions": {
            "train": float(object_split_fractions[0]),
            "validation": float(object_split_fractions[1]),
            "test": float(object_split_fractions[2]),
        },
        "min_episode_steps": int(min_episode_steps),
        "objects": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    validate_manifest(output, inspect_datasets=True)
    return manifest


def _resolve_optional_path(value: Any, parent: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _resolve_required_path(value: Any, parent: Path) -> Path:
    path = _resolve_optional_path(value, parent)
    if path is None:
        raise ValueError("Required path value is empty")
    return path


def load_manifest(path: Path) -> tuple[dict[str, Any], tuple[ObjectSpec, ...]]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version={payload.get('schema_version')}; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )
    objects: list[ObjectSpec] = []
    for record in payload.get("objects", []):
        objects.append(
            ObjectSpec(
                object_id=str(record["object_id"]),
                physical_config_id=str(record["physical_config_id"]),
                shape_id=str(record["shape_id"]),
                trajectory_npz=_resolve_required_path(record["trajectory_npz"], manifest_path.parent),
                dino_feature_npz=_resolve_optional_path(record.get("dino_feature_npz"), manifest_path.parent),
                object_split=str(record["object_split"]),
                context_episode_indices=tuple(int(value) for value in record["context_episode_indices"]),
                query_episode_indices=tuple(int(value) for value in record["query_episode_indices"]),
                eval_episode_indices=tuple(int(value) for value in record["eval_episode_indices"]),
                friction_spec=dict(record.get("friction_spec", {})),
                num_trajectories=int(record["num_trajectories"]),
                min_steps=int(record["min_steps"]),
                max_steps=int(record["max_steps"]),
            )
        )
    return payload, tuple(objects)


def validate_manifest(path: Path, *, inspect_datasets: bool = True) -> dict[str, Any]:
    payload, objects = load_manifest(path)
    if not objects:
        raise ValueError("Manifest contains no objects")
    object_ids = [obj.object_id for obj in objects]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Manifest object_id values must be unique")

    errors: list[str] = []
    split_counts = {split: 0 for split in VALID_OBJECT_SPLITS}
    for obj in objects:
        if obj.object_split not in VALID_OBJECT_SPLITS:
            errors.append(f"{obj.object_id}: invalid object_split={obj.object_split!r}")
        else:
            split_counts[obj.object_split] += 1
        pools = {pool: set(obj.pool_indices(pool)) for pool in VALID_EPISODE_POOLS}
        if any(not indices for indices in pools.values()):
            errors.append(f"{obj.object_id}: context/query/eval pools must all be non-empty")
        if pools["context"] & pools["query"] or pools["context"] & pools["eval"] or pools["query"] & pools["eval"]:
            errors.append(f"{obj.object_id}: context/query/eval pools overlap")
        all_indices = pools["context"] | pools["query"] | pools["eval"]
        if any(index < 0 or index >= obj.num_trajectories for index in all_indices):
            errors.append(f"{obj.object_id}: episode index is outside [0, {obj.num_trajectories - 1}]")
        if inspect_datasets:
            try:
                header = inspect_dataset_header(obj.trajectory_npz, min_episode_steps=1)
            except Exception as exc:
                errors.append(f"{obj.object_id}: dataset inspection failed: {exc}")
            else:
                if header.num_trajectories != obj.num_trajectories:
                    errors.append(
                        f"{obj.object_id}: manifest num_trajectories={obj.num_trajectories} "
                        f"but dataset has {header.num_trajectories}"
                    )
        if obj.dino_feature_npz is not None and not obj.dino_feature_npz.is_file():
            errors.append(f"{obj.object_id}: DINO feature file does not exist: {obj.dino_feature_npz}")

    config_splits: dict[str, set[str]] = {}
    for obj in objects:
        config_splits.setdefault(obj.physical_config_id, set()).add(obj.object_split)
    leaking_configs = sorted(config_id for config_id, splits in config_splits.items() if len(splits) > 1)
    if leaking_configs:
        errors.append(f"physical_config_id values cross object splits: {leaking_configs}")
    if errors:
        raise ValueError("Manifest validation failed:\n- " + "\n- ".join(errors))
    return {
        "schema_version": int(payload["schema_version"]),
        "objects": len(objects),
        "object_split_counts": split_counts,
        "physical_configs": len(config_splits),
        "episode_counts": {
            pool: int(sum(len(obj.pool_indices(pool)) for obj in objects)) for pool in VALID_EPISODE_POOLS
        },
    }


def quaternion_xyzw_to_yaw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1, 4)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norms, 1.0e-8)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)


def trajectory_to_encoder_features(trajectory: MujocoTrajectory) -> np.ndarray:
    num_steps = trajectory.num_steps
    if num_steps < 1:
        raise ValueError("Encoder trajectory must contain at least one step")

    yaw = quaternion_xyzw_to_yaw(trajectory.quaternions_xyzw[:num_steps])
    yaw0 = float(yaw[0])
    cosine = float(np.cos(yaw0))
    sine = float(np.sin(yaw0))
    world_to_initial = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float32)

    relative_xy_world = trajectory.positions[:num_steps, :2] - trajectory.positions[0, :2]
    relative_xy_local = relative_xy_world @ world_to_initial.T
    velocity_local = trajectory.linear_velocity[:num_steps, :2] @ world_to_initial.T
    force_local = trajectory.step_forces[:num_steps, :2] @ world_to_initial.T
    relative_yaw = np.unwrap(yaw.astype(np.float64) - yaw0).astype(np.float32)
    force_point = np.broadcast_to(
        np.asarray(trajectory.force_point_offset_local[:2], dtype=np.float32),
        (num_steps, 2),
    )
    force_magnitude = np.linalg.norm(trajectory.step_forces[:num_steps], axis=1, keepdims=True).astype(np.float32)

    features = np.concatenate(
        [
            relative_xy_local,
            np.sin(relative_yaw)[:, None],
            np.cos(relative_yaw)[:, None],
            velocity_local,
            trajectory.angular_velocity[:num_steps, 2:3],
            force_local,
            force_point,
            force_magnitude,
        ],
        axis=1,
    ).astype(np.float32)
    if features.shape[1] != len(ENCODER_FEATURE_SCHEMA):
        raise RuntimeError(f"Encoder feature width mismatch: {features.shape[1]} vs {len(ENCODER_FEATURE_SCHEMA)}")
    if not np.all(np.isfinite(features)):
        raise ValueError("Encoder features contain non-finite values")
    return features


def build_encoder_feature_batch(
    trajectories: Sequence[MujocoTrajectory],
    *,
    episode_indices: Sequence[int],
    window_start_steps: Sequence[int],
) -> EncoderFeatureBatch:
    if not trajectories:
        raise ValueError("Cannot build an encoder feature batch from no trajectories")
    feature_arrays = [trajectory_to_encoder_features(trajectory) for trajectory in trajectories]
    lengths = np.asarray([len(features) for features in feature_arrays], dtype=np.int32)
    max_steps = int(np.max(lengths))
    features = np.zeros((len(feature_arrays), max_steps, len(ENCODER_FEATURE_SCHEMA)), dtype=np.float32)
    valid_mask = np.zeros((len(feature_arrays), max_steps), dtype=np.bool_)
    for index, values in enumerate(feature_arrays):
        features[index, : len(values)] = values
        valid_mask[index, : len(values)] = True
    return EncoderFeatureBatch(
        features=features,
        valid_mask=valid_mask,
        lengths=lengths,
        episode_indices=np.asarray(episode_indices, dtype=np.int32),
        window_start_steps=np.asarray(window_start_steps, dtype=np.int32),
    )


class ObjectPhysicsDataset:
    def __init__(
        self,
        manifest_path: Path,
        *,
        cache_size: int | None = 8,
        load_max_steps: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest, objects = load_manifest(self.manifest_path)
        validate_manifest(self.manifest_path, inspect_datasets=False)
        self.objects = {obj.object_id: obj for obj in objects}
        self.cache_size = None if cache_size is None or int(cache_size) <= 0 else int(cache_size)
        self.load_max_steps = load_max_steps
        self._collections: OrderedDict[str, MujocoTrajectoryCollection] = OrderedDict()

    def object_ids(self, split: str | None = None) -> tuple[str, ...]:
        if split is not None and split not in VALID_OBJECT_SPLITS:
            raise ValueError(f"Unknown object split {split!r}; expected one of {VALID_OBJECT_SPLITS}")
        return tuple(
            sorted(obj.object_id for obj in self.objects.values() if split is None or obj.object_split == split)
        )

    def get_object(self, object_id: str) -> ObjectSpec:
        try:
            return self.objects[str(object_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown object_id {object_id!r}") from exc

    def load_object_collection(self, object_id: str) -> MujocoTrajectoryCollection:
        key = str(object_id)
        if key in self._collections:
            collection = self._collections.pop(key)
            self._collections[key] = collection
            return collection
        obj = self.get_object(key)
        collection = load_mujoco_trajectories(obj.trajectory_npz, max_steps=self.load_max_steps)
        self._collections[key] = collection
        if self.cache_size is not None:
            while len(self._collections) > self.cache_size:
                self._collections.popitem(last=False)
        return collection

    def sample_object_ids(
        self,
        *,
        split: str,
        count: int,
        rng: np.random.Generator,
    ) -> tuple[str, ...]:
        candidates = self.object_ids(split)
        if not candidates:
            raise ValueError(f"No objects are available in split {split!r}")
        if int(count) > len(candidates):
            raise ValueError(
                f"Requested {int(count)} distinct objects from split {split!r}, "
                f"but only {len(candidates)} are available"
            )
        selected = rng.choice(np.asarray(candidates, dtype=object), size=int(count), replace=False)
        return tuple(str(value) for value in selected.tolist())

    @staticmethod
    def _sample_disjoint_indices(
        pool: Sequence[int],
        *,
        sizes: Sequence[int],
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        total = int(sum(int(size) for size in sizes))
        values = np.asarray(pool, dtype=np.int32)
        if total > len(values):
            raise ValueError(f"Requested {total} distinct episodes from a pool containing {len(values)} episodes")
        selected = values[rng.choice(len(values), size=total, replace=False)]
        result: list[np.ndarray] = []
        cursor = 0
        for size in sizes:
            end = cursor + int(size)
            result.append(np.asarray(selected[cursor:end], dtype=np.int32))
            cursor = end
        return result

    @staticmethod
    def _select_windows(
        collection: MujocoTrajectoryCollection,
        episode_indices: Sequence[int],
        *,
        window_steps: int | None,
        random_time_windows: bool,
        rng: np.random.Generator,
    ) -> tuple[tuple[MujocoTrajectory, ...], np.ndarray]:
        trajectory_by_episode_index = {
            int(trajectory.metadata.get("episode_index", loaded_index)): trajectory
            for loaded_index, trajectory in enumerate(collection.trajectories)
        }
        trajectories: list[MujocoTrajectory] = []
        start_steps: list[int] = []
        for episode_index in episode_indices:
            if int(episode_index) not in trajectory_by_episode_index:
                raise KeyError(f"Loaded trajectory collection does not contain episode_index={int(episode_index)}")
            trajectory = trajectory_by_episode_index[int(episode_index)]
            if window_steps is None:
                selected = trajectory
                start_step = 0
            elif random_time_windows:
                selected, start_step = sample_mujoco_trajectory_time_window(
                    trajectory,
                    window_steps=int(window_steps),
                    rng=rng,
                )
            else:
                selected = slice_mujoco_trajectory_time_window(
                    trajectory,
                    start_step=0,
                    window_steps=int(window_steps),
                )
                start_step = 0
            trajectories.append(selected)
            start_steps.append(int(start_step))
        return tuple(trajectories), np.asarray(start_steps, dtype=np.int32)

    def sample_object_training_data(
        self,
        object_id: str,
        *,
        context_trajectories_per_view: int,
        query_trajectories_per_view: int,
        context_window_steps: int | None,
        query_window_steps: int | None,
        random_context_windows: bool,
        random_query_windows: bool,
        rng: np.random.Generator,
    ) -> ObjectTrainingSample:
        obj = self.get_object(object_id)
        collection = self.load_object_collection(object_id)
        context_a_idx, context_b_idx = self._sample_disjoint_indices(
            obj.context_episode_indices,
            sizes=(context_trajectories_per_view, context_trajectories_per_view),
            rng=rng,
        )
        query_a_idx, query_b_idx = self._sample_disjoint_indices(
            obj.query_episode_indices,
            sizes=(query_trajectories_per_view, query_trajectories_per_view),
            rng=rng,
        )
        context_a_trajectories, context_a_starts = self._select_windows(
            collection,
            context_a_idx,
            window_steps=context_window_steps,
            random_time_windows=random_context_windows,
            rng=rng,
        )
        context_b_trajectories, context_b_starts = self._select_windows(
            collection,
            context_b_idx,
            window_steps=context_window_steps,
            random_time_windows=random_context_windows,
            rng=rng,
        )
        query_a_trajectories, query_a_starts = self._select_windows(
            collection,
            query_a_idx,
            window_steps=query_window_steps,
            random_time_windows=random_query_windows,
            rng=rng,
        )
        query_b_trajectories, query_b_starts = self._select_windows(
            collection,
            query_b_idx,
            window_steps=query_window_steps,
            random_time_windows=random_query_windows,
            rng=rng,
        )
        return ObjectTrainingSample(
            object_spec=obj,
            context_a=build_encoder_feature_batch(
                context_a_trajectories,
                episode_indices=context_a_idx,
                window_start_steps=context_a_starts,
            ),
            context_b=build_encoder_feature_batch(
                context_b_trajectories,
                episode_indices=context_b_idx,
                window_start_steps=context_b_starts,
            ),
            query_a=QueryTrajectoryBatch(
                trajectories=query_a_trajectories,
                episode_indices=query_a_idx,
                window_start_steps=query_a_starts,
            ),
            query_b=QueryTrajectoryBatch(
                trajectories=query_b_trajectories,
                episode_indices=query_b_idx,
                window_start_steps=query_b_starts,
            ),
        )

    def sample_training_step(
        self,
        *,
        split: str,
        objects_per_step: int,
        context_trajectories_per_view: int,
        query_trajectories_per_view: int,
        context_window_steps: int | None,
        query_window_steps: int | None,
        random_context_windows: bool = True,
        random_query_windows: bool = True,
        rng: np.random.Generator,
    ) -> tuple[ObjectTrainingSample, ...]:
        object_ids = self.sample_object_ids(split=split, count=objects_per_step, rng=rng)
        return tuple(
            self.sample_object_training_data(
                object_id,
                context_trajectories_per_view=context_trajectories_per_view,
                query_trajectories_per_view=query_trajectories_per_view,
                context_window_steps=context_window_steps,
                query_window_steps=query_window_steps,
                random_context_windows=random_context_windows,
                random_query_windows=random_query_windows,
                rng=rng,
            )
            for object_id in object_ids
        )


def discover_npz_datasets(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_file():
            paths.add(path.resolve())
        elif path.is_dir():
            for candidate in path.rglob("*.npz"):
                try:
                    with np.load(candidate, allow_pickle=True) as data:
                        is_dataset = {"trajectories", "columns", "episode_lengths"}.issubset(data.files)
                except Exception:
                    is_dataset = False
                if is_dataset:
                    paths.add(candidate.resolve())
        else:
            raise FileNotFoundError(f"Dataset input does not exist: {path}")
    return sorted(paths)
