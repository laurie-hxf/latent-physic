from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
NEWTON_DIR = REPO_ROOT / "newton"
for _path in (REPO_ROOT, NEWTON_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from fit_mujoco_contact_point_friction_io import save_contact_friction_point_cloud  # noqa: E402
from fit_mujoco_contact_point_friction_kernels import (  # noqa: E402
    accumulate_batched_frame_loss_kernel,
    apply_batched_external_and_surface_point_forces_trajectory_kernel,
    combine_batched_loss_components_kernel,
    compute_batched_contact_weighted_masses_kernel,
    scatter_active_point_friction_kernel,
    sum_batched_losses_kernel,
)
from fit_mujoco_contact_point_friction_params import compute_piecewise_side_ids  # noqa: E402
from fit_mujoco_contact_point_friction_runtime import (  # noqa: E402
    build_batched_optimization_buffers,
    clear_batched_optimization_grads,
    forward_rollout_with_batched_trajectory_loss,
    resolve_point_position_loss_scale,
    reset_scene_states,
    set_batched_box_initial_states_kernel,
)
from mujoco_contact_friction_fit_utils import (  # noqa: E402
    MujocoTrajectory,
    compute_active_contact_point_indices,
)
from newton_surface_points_diff_demo import GRAVITY_MAGNITUDE, _smoothstep01, build_diff_scene  # noqa: E402
from object_physics_latent.dataset import (  # noqa: E402
    ENCODER_FEATURE_SCHEMA,
    EncoderFeatureBatch,
    ObjectPhysicsDataset,
    ObjectSpec,
)
from object_physics_latent.encoder import latent_regularization_losses  # noqa: E402
from object_physics_latent.friction_decoder import build_point_conditioning_features  # noqa: E402
from object_physics_latent.model import TrajectoryConditionedFrictionModel  # noqa: E402


DEFAULT_MANIFEST = (
    REPO_ROOT / "mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/manifest.json"
)
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "outputs/object_physics_latent_dino_mlp"


@dataclass(frozen=True)
class PointFeatureEntry:
    object_id: str
    features: torch.Tensor
    visual_features: torch.Tensor | None
    metadata: dict
    stats: dict[str, np.ndarray]


@dataclass(frozen=True)
class RolloutDiagnostics:
    loss: float
    position_loss: float
    orientation_loss: float
    linear_velocity_loss: float
    angular_velocity_loss: float
    grad_norm: float
    grad_abs_mean: float
    grad_abs_max: float
    mu_mean: float
    mu_std: float
    mu_min: float
    mu_max: float


@dataclass(frozen=True)
class ViewRolloutPlan:
    object_index: int
    object_id: str
    view_name: str
    start: int
    end: int
    latent: torch.Tensor
    active_mu: torch.Tensor


@dataclass(frozen=True)
class SwapRolloutPlan:
    object_index: int
    object_id: str
    negative_object_id: str
    view_name: str
    positive_start: int
    positive_end: int
    negative_start: int
    negative_end: int
    positive_mu: torch.Tensor
    negative_mu: torch.Tensor


class PointFeatureCache:
    def __init__(
        self,
        *,
        diff_scene,
        args: argparse.Namespace,
        torch_device: torch.device,
    ) -> None:
        self.diff_scene = diff_scene
        self.args = args
        self.torch_device = torch_device
        self._cache: dict[str, PointFeatureEntry] = {}
        self.input_dim: int | None = None
        self.visual_input_dim: int | None = None
        self.reference_metadata: dict | None = None

    def get(self, obj: ObjectSpec) -> PointFeatureEntry:
        object_id = str(obj.object_id)
        if object_id in self._cache:
            return self._cache[object_id]

        if bool(self.args.no_dino):
            dino_path = None
        else:
            dino_path = obj.dino_feature_npz
            if dino_path is None:
                raise ValueError(f"{object_id} does not have dino_feature_npz in the manifest.")
            if not Path(dino_path).is_file():
                raise FileNotFoundError(f"{object_id} DINO feature file does not exist: {dino_path}")

        features_np, metadata, stats = build_point_conditioning_features(
            local_surface_points=self.diff_scene.local_surface_points_np,
            half_extents=np.asarray(self.args.box_half_extents, dtype=np.float32),
            dino_npz_path=dino_path,
            position_frequencies=int(self.args.dino_position_frequencies),
            neighbor_radius=float(self.args.dino_neighbor_radius),
            neighbor_k=int(self.args.dino_neighbor_k),
            normalize_dino=bool(self.args.dino_feature_normalization),
            max_match_distance=float(self.args.dino_mlp_max_match_distance),
        )
        if not np.all(np.isfinite(features_np)):
            raise ValueError(f"{object_id} point conditioning features contain non-finite values.")

        full_feature_dim = int(features_np.shape[1])
        encoded_position_dim = int(metadata.encoded_position_dim)
        feature_mode = str(getattr(self.args, "decoder_point_feature_mode", "full"))
        if feature_mode == "position":
            decoder_features_np = features_np[:, :encoded_position_dim]
        elif feature_mode == "full":
            decoder_features_np = features_np
        else:
            raise ValueError(f"Unknown decoder point feature mode: {feature_mode!r}")
        visual_features_np = (
            features_np
            if bool(getattr(self.args, "dino_to_encoder", False)) and int(metadata.dino_dim) > 0
            else None
        )

        feature_dim = int(decoder_features_np.shape[1])
        visual_feature_dim = 0 if visual_features_np is None else int(visual_features_np.shape[1])
        if self.input_dim is None:
            self.input_dim = feature_dim
            self.visual_input_dim = visual_feature_dim
            self.reference_metadata = {
                "input_dim": int(feature_dim),
                "decoder_input_dim": int(feature_dim),
                "visual_input_dim": int(visual_feature_dim),
                "full_input_dim": int(full_feature_dim),
                "decoder_point_feature_mode": str(feature_mode),
                "dino_to_encoder": bool(getattr(self.args, "dino_to_encoder", False)),
                "encoded_position_dim": int(metadata.encoded_position_dim),
                "dino_dim": int(metadata.dino_dim),
                "position_frequencies": int(metadata.position_frequencies),
                "neighbor_radius": float(metadata.neighbor_radius),
                "neighbor_k": int(metadata.neighbor_k),
                "normalize_dino": bool(metadata.normalize_dino),
            }
        elif feature_dim != self.input_dim or visual_feature_dim != self.visual_input_dim:
            raise ValueError(
                f"{object_id} decoder/visual point feature dim={feature_dim}/{visual_feature_dim}, "
                f"expected {self.input_dim}/{self.visual_input_dim}. "
                "All objects in one training run must share the same DINO/position feature width."
            )

        entry = PointFeatureEntry(
            object_id=object_id,
            features=torch.as_tensor(decoder_features_np, dtype=torch.float32, device=self.torch_device),
            visual_features=None
            if visual_features_np is None
            else torch.as_tensor(visual_features_np, dtype=torch.float32, device=self.torch_device),
            metadata={
                "input_dim": int(feature_dim),
                "decoder_input_dim": int(feature_dim),
                "visual_input_dim": int(visual_feature_dim),
                "full_input_dim": int(full_feature_dim),
                "decoder_point_feature_mode": str(feature_mode),
                "dino_to_encoder": bool(getattr(self.args, "dino_to_encoder", False)),
                "encoded_position_dim": int(metadata.encoded_position_dim),
                "dino_dim": int(metadata.dino_dim),
                "position_frequencies": int(metadata.position_frequencies),
                "neighbor_radius": float(metadata.neighbor_radius),
                "neighbor_k": int(metadata.neighbor_k),
                "normalize_dino": bool(metadata.normalize_dino),
            },
            stats=stats,
        )
        self._cache[object_id] = entry
        return entry


class ObjectBatchSampler:
    """Samples object ids for one optimizer step.

    ``round_robin`` keeps the long-run schedule balanced: each shuffled epoch
    contains every object exactly once, while still avoiding duplicates inside
    a single optimizer step.
    """

    def __init__(
        self,
        object_ids: Sequence[str],
        *,
        objects_per_step: int,
        strategy: str,
        rng: np.random.Generator,
    ) -> None:
        self.object_ids = tuple(str(value) for value in object_ids)
        self.objects_per_step = int(objects_per_step)
        self.strategy = str(strategy)
        self.rng = rng
        self._queue: list[str] = []
        self.epoch_index = 0

        if not self.object_ids:
            raise ValueError("ObjectBatchSampler requires at least one object.")
        if self.objects_per_step < 1:
            raise ValueError("objects_per_step must be >= 1.")
        if self.objects_per_step > len(self.object_ids):
            raise ValueError(
                f"objects_per_step={self.objects_per_step} exceeds object_count={len(self.object_ids)}."
            )
        if self.strategy not in {"random", "round_robin"}:
            raise ValueError(f"Unknown object sampling strategy: {self.strategy!r}")

    def _refill_round_robin_queue(self, *, selected_this_step: set[str]) -> None:
        order = list(self.object_ids)
        self.rng.shuffle(order)
        if selected_this_step:
            # If a batch crosses an epoch boundary, delay already-selected
            # objects so the same object does not appear twice in one step.
            order = [value for value in order if value not in selected_this_step] + [
                value for value in order if value in selected_this_step
            ]
        self._queue.extend(order)
        self.epoch_index += 1

    def sample(self) -> tuple[str, ...]:
        if self.strategy == "random":
            selected = self.rng.choice(
                np.asarray(self.object_ids, dtype=object),
                size=self.objects_per_step,
                replace=False,
            )
            return tuple(str(value) for value in selected.tolist())

        selected: list[str] = []
        selected_set: set[str] = set()
        while len(selected) < self.objects_per_step:
            if not self._queue:
                self._refill_round_robin_queue(selected_this_step=selected_set)
            object_id = self._queue.pop(0)
            if object_id in selected_set:
                self._queue.append(object_id)
                continue
            selected.append(object_id)
            selected_set.add(object_id)
        return tuple(selected)

    def state_dict(self) -> dict:
        return {
            "object_ids": list(self.object_ids),
            "objects_per_step": int(self.objects_per_step),
            "strategy": str(self.strategy),
            "queue": list(self._queue),
            "epoch_index": int(self.epoch_index),
        }

    def load_state_dict(self, state: dict) -> None:
        saved_object_ids = tuple(str(value) for value in state["object_ids"])
        if saved_object_ids != self.object_ids:
            raise ValueError("Resume checkpoint object ids do not match the current object split.")
        if int(state["objects_per_step"]) != self.objects_per_step:
            raise ValueError("Resume checkpoint objects_per_step does not match the current run.")
        if str(state["strategy"]) != self.strategy:
            raise ValueError("Resume checkpoint object sampling strategy does not match the current run.")
        queue = [str(value) for value in state.get("queue", [])]
        unknown = sorted(set(queue) - set(self.object_ids))
        if unknown:
            raise ValueError(f"Resume checkpoint sampler queue contains unknown object ids: {unknown}")
        self._queue = queue
        self.epoch_index = int(state.get("epoch_index", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--object-split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch-device", type=str, default=None)
    parser.add_argument("--opt-iters", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--best-checkpoint-every",
        type=int,
        default=1,
        help=(
            "Save the best checkpoint to disk every N best updates. Values >1 avoid frequent disk "
            "serialization; the exact latest best checkpoint is still written at training end."
        ),
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume full training state. --opt-iters remains the final target iteration.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="object-physics-latent")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-resume-id", type=str, default=None)
    parser.add_argument(
        "--wandb-log-detail",
        choices=("summary", "full"),
        default="full",
        help="Use summary to skip high-cardinality per-object/per-view W&B metrics.",
    )

    parser.add_argument("--objects-per-step", type=int, default=2)
    parser.add_argument(
        "--object-sampling-strategy",
        choices=("random", "round_robin"),
        default="round_robin",
        help="How to choose the object ids used by each optimizer step.",
    )
    parser.add_argument("--context-trajectories-per-view", type=int, default=4)
    parser.add_argument("--query-trajectories-per-view", type=int, default=64)
    parser.add_argument("--context-window-steps", type=int, default=300)
    parser.add_argument("--query-window-steps", type=int, default=300)
    parser.add_argument("--no-random-context-windows", dest="random_context_windows", action="store_false", default=True)
    parser.add_argument("--no-random-query-windows", dest="random_query_windows", action="store_false", default=True)
    parser.add_argument(
        "--time-window-source-max-steps",
        type=int,
        default=None,
        help="Optional source-trajectory truncation before dataset window sampling.",
    )
    parser.add_argument("--dataset-cache-size", type=int, default=4)

    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--step-hidden-dim", type=int, default=128)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--trajectory-embedding-dim", type=int, default=128)
    parser.add_argument("--set-hidden-dim", type=int, default=128)
    parser.add_argument("--visual-hidden-dim", type=int, default=128)
    parser.add_argument("--visual-embedding-dim", type=int, default=128)
    parser.add_argument("--visual-point-hidden-layers", type=int, default=1)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-hidden-layers", type=int, default=2)
    parser.add_argument("--decoder-conditioning", choices=("concat", "film", "basis"), default="film")
    parser.add_argument("--decoder-activation", choices=("relu", "silu"), default="silu")
    parser.add_argument(
        "--decoder-basis-count",
        type=int,
        default=8,
        help="Number of spatial basis functions used when --decoder-conditioning=basis.",
    )
    parser.add_argument(
        "--decoder-basis-base-mode",
        choices=("latent", "global_shared", "fixed"),
        default="latent",
        help=(
            "Base raw-logit route for basis decoder. Use latent for legacy BaseHead(z), "
            "global_shared for one learned scalar shared by all objects, or fixed for a constant initial base."
        ),
    )
    parser.add_argument(
        "--decoder-basis-normalization",
        choices=("none", "zero_mean", "unit_std"),
        default="zero_mean",
        help="Point-wise normalization for basis functions before multiplying by latent coefficients.",
    )
    parser.add_argument(
        "--decoder-basis-activation",
        choices=("tanh", "identity"),
        default="tanh",
        help="Activation applied to raw basis functions before point-wise normalization.",
    )
    parser.add_argument(
        "--decoder-basis-norm-eps",
        type=float,
        default=1.0e-4,
        help="Minimum point-wise std used by --decoder-basis-normalization=unit_std.",
    )
    parser.add_argument(
        "--decoder-latent-normalization",
        choices=("none", "layernorm"),
        default="none",
        help="Normalize object latent before it is consumed by the friction decoder.",
    )
    parser.add_argument(
        "--decoder-raw-limit",
        type=float,
        default=None,
        help="Optional tanh limit on decoder raw logits before the bounded friction sigmoid.",
    )
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument(
        "--latent-norm-weight",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option. The encoder now emits unit-norm "
            "latents directly, so this weight is not added to the objective."
        ),
    )
    parser.add_argument("--latent-norm-target", type=float, default=8.0)
    parser.add_argument("--swap-loss-weight", type=float, default=0.0)
    parser.add_argument("--swap-loss-margin", type=float, default=0.02)
    parser.add_argument("--swap-loss-temperature", type=float, default=0.05)
    parser.add_argument("--swap-query-trajectories-per-view", type=int, default=64)

    parser.add_argument("--no-dino", action="store_true", help="Use position-only point features.")
    parser.add_argument(
        "--decoder-point-feature-mode",
        choices=("full", "position"),
        default="position",
        help="Features exposed directly to the friction decoder. Use position to keep DINO out of the decoder.",
    )
    parser.add_argument(
        "--dino-to-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse object-level DINO/point-set features into the trajectory latent instead of feeding DINO to the decoder.",
    )
    parser.add_argument("--dino-neighbor-radius", type=float, default=0.025)
    parser.add_argument("--dino-neighbor-k", type=int, default=16)
    parser.add_argument("--dino-position-frequencies", type=int, default=6)
    parser.add_argument("--dino-mlp-max-match-distance", type=float, default=1.0e-5)
    parser.add_argument(
        "--no-dino-feature-normalization",
        dest="dino_feature_normalization",
        action="store_false",
        default=True,
    )

    parser.add_argument("--min-point-friction", type=float, default=0.0)
    parser.add_argument("--max-point-friction", type=float, default=2.0)
    parser.add_argument("--point-friction", type=float, default=0.35)
    parser.add_argument("--contact-friction", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--friction-regularization", type=float, default=1.0e-3)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--contact-mask-threshold", type=float, default=0.002)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=(0.58, 0.0, 0.025))
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--avoid-zero-surface-point-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=1.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.1)
    parser.add_argument("--point-position-loss-reduction", choices=("sum", "mean"), default="mean")

    parser.add_argument("--active-object-limit", type=int, default=4)
    parser.add_argument("--active-trajectories-per-object", type=int, default=64)
    parser.add_argument("--active-window-steps", type=int, default=None)
    parser.add_argument("--active-use-query-pool", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--export-preview-count", type=int, default=4)
    parser.add_argument("--point-cloud-color-min", type=float, default=None)
    parser.add_argument("--point-cloud-color-max", type=float, default=None)
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    return value


def _args_dict(args: argparse.Namespace) -> dict:
    return {key: _jsonable(value) for key, value in vars(args).items()}


def log(message: str) -> None:
    print(message, flush=True)


def validate_args(args: argparse.Namespace) -> None:
    if int(args.opt_iters) < 1 and not bool(args.dry_run):
        raise ValueError("--opt-iters must be >= 1")
    if int(args.best_checkpoint_every) < 1:
        raise ValueError("--best-checkpoint-every must be >= 1")
    if int(args.objects_per_step) < 1:
        raise ValueError("--objects-per-step must be >= 1")
    if int(args.context_trajectories_per_view) < 1:
        raise ValueError("--context-trajectories-per-view must be >= 1")
    if int(args.query_trajectories_per_view) < 1:
        raise ValueError("--query-trajectories-per-view must be >= 1")
    if int(args.context_window_steps) < 1:
        raise ValueError("--context-window-steps must be >= 1")
    if int(args.query_window_steps) < 1:
        raise ValueError("--query-window-steps must be >= 1")
    if float(args.max_point_friction) <= float(args.min_point_friction):
        raise ValueError("--max-point-friction must be greater than --min-point-friction")
    if not (float(args.min_point_friction) <= float(args.point_friction) <= float(args.max_point_friction)):
        raise ValueError("--point-friction must lie inside [--min-point-friction, --max-point-friction]")
    if int(args.decoder_basis_count) < 1:
        raise ValueError("--decoder-basis-count must be >= 1")
    if float(args.decoder_basis_norm_eps) <= 0.0:
        raise ValueError("--decoder-basis-norm-eps must be > 0")
    if args.decoder_raw_limit is not None and float(args.decoder_raw_limit) <= 0.0:
        raise ValueError("--decoder-raw-limit must be > 0 when provided")
    if float(args.latent_norm_weight) < 0.0:
        raise ValueError("--latent-norm-weight must be >= 0")
    if float(args.latent_norm_target) <= 0.0:
        raise ValueError("--latent-norm-target must be > 0")
    if float(args.swap_loss_weight) < 0.0:
        raise ValueError("--swap-loss-weight must be >= 0")
    if float(args.swap_loss_margin) < 0.0:
        raise ValueError("--swap-loss-margin must be >= 0")
    if float(args.swap_loss_temperature) <= 0.0:
        raise ValueError("--swap-loss-temperature must be > 0")
    if int(args.swap_query_trajectories_per_view) < 1:
        raise ValueError("--swap-query-trajectories-per-view must be >= 1")
    if float(args.swap_loss_weight) > 0.0 and int(args.objects_per_step) < 2:
        raise ValueError("--swap-loss-weight > 0 requires --objects-per-step >= 2")
    if bool(args.no_dino) and bool(args.dino_to_encoder):
        args.dino_to_encoder = False
    if args.resume_checkpoint is not None and not Path(args.resume_checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"--resume-checkpoint does not exist: {args.resume_checkpoint}")


def stack_encoder_batches(
    batches: Iterable[EncoderFeatureBatch],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_list = list(batches)
    if not batch_list:
        raise ValueError("Cannot stack an empty encoder batch list.")
    object_count = len(batch_list)
    trajectories_per_object = max(int(batch.features.shape[0]) for batch in batch_list)
    max_steps = max(int(batch.features.shape[1]) for batch in batch_list)
    feature_dim = int(batch_list[0].features.shape[2])
    features = np.zeros((object_count, trajectories_per_object, max_steps, feature_dim), dtype=np.float32)
    valid_mask = np.zeros((object_count, trajectories_per_object, max_steps), dtype=np.bool_)
    for object_idx, batch in enumerate(batch_list):
        if int(batch.features.shape[2]) != feature_dim:
            raise ValueError(f"Encoder feature dim mismatch: {batch.features.shape[2]} vs {feature_dim}")
        traj_count, steps, _ = batch.features.shape
        features[object_idx, :traj_count, :steps] = batch.features
        valid_mask[object_idx, :traj_count, :steps] = batch.valid_mask
    return (
        torch.as_tensor(features, dtype=torch.float32, device=device),
        torch.as_tensor(valid_mask, dtype=torch.bool, device=device),
    )


def stack_visual_features(entries: Iterable[PointFeatureEntry], *, device: torch.device) -> torch.Tensor | None:
    entry_list = list(entries)
    if not entry_list:
        raise ValueError("Cannot stack visual features for an empty entry list.")
    visual_items = [entry.visual_features for entry in entry_list]
    if all(item is None for item in visual_items):
        return None
    if any(item is None for item in visual_items):
        raise ValueError("Visual features are missing for some objects but present for others.")
    return torch.stack([item.to(device=device) for item in visual_items if item is not None], dim=0)


def sample_training_data_for_object_ids(
    *,
    dataset: ObjectPhysicsDataset,
    object_ids: Sequence[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple:
    return tuple(
        dataset.sample_object_training_data(
            str(object_id),
            context_trajectories_per_view=int(args.context_trajectories_per_view),
            query_trajectories_per_view=int(args.query_trajectories_per_view),
            context_window_steps=int(args.context_window_steps),
            query_window_steps=int(args.query_window_steps),
            random_context_windows=bool(args.random_context_windows),
            random_query_windows=bool(args.random_query_windows),
            rng=rng,
        )
        for object_id in object_ids
    )


def summarize_object_sampling(
    *,
    all_object_ids: Sequence[str],
    selected_object_ids: Sequence[str],
    appearance_counts: dict[str, int],
    iteration: int,
    objects_per_step: int,
) -> dict[str, float | int]:
    counts = np.asarray([int(appearance_counts.get(str(object_id), 0)) for object_id in all_object_ids], dtype=np.float64)
    selected_counts = np.asarray(
        [int(appearance_counts.get(str(object_id), 0)) for object_id in selected_object_ids],
        dtype=np.float64,
    )
    expected = float(iteration) * float(objects_per_step) / max(float(len(all_object_ids)), 1.0)
    result: dict[str, float | int] = {
        "min_appearance_count": int(np.min(counts)) if counts.size else 0,
        "max_appearance_count": int(np.max(counts)) if counts.size else 0,
        "mean_appearance_count": float(np.mean(counts)) if counts.size else 0.0,
        "std_appearance_count": float(np.std(counts)) if counts.size else 0.0,
        "expected_appearance_count": expected,
        "selected_min_appearance_count": int(np.min(selected_counts)) if selected_counts.size else 0,
        "selected_max_appearance_count": int(np.max(selected_counts)) if selected_counts.size else 0,
        "selected_mean_appearance_count": float(np.mean(selected_counts)) if selected_counts.size else 0.0,
    }
    return result


def select_trajectories_for_active_mask(
    *,
    dataset: ObjectPhysicsDataset,
    object_ids: tuple[str, ...],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[MujocoTrajectory]:
    selected: list[MujocoTrajectory] = []
    object_limit = min(len(object_ids), max(int(args.active_object_limit), 1))
    for object_id in object_ids[:object_limit]:
        obj = dataset.get_object(object_id)
        pool = obj.query_episode_indices if bool(args.active_use_query_pool) else obj.context_episode_indices
        if not pool:
            continue
        collection = dataset.load_object_collection(object_id)
        by_episode = {
            int(trajectory.metadata.get("episode_index", loaded_idx)): trajectory
            for loaded_idx, trajectory in enumerate(collection.trajectories)
        }
        pool_values = np.asarray(pool, dtype=np.int32)
        take_count = min(len(pool_values), max(int(args.active_trajectories_per_object), 1))
        if take_count < len(pool_values):
            chosen = pool_values[rng.choice(len(pool_values), size=take_count, replace=False)]
        else:
            chosen = pool_values
        for episode_idx in chosen:
            trajectory = by_episode.get(int(episode_idx))
            if trajectory is None:
                continue
            if args.active_window_steps is not None:
                from mujoco_contact_friction_fit_utils import slice_mujoco_trajectory_time_window

                trajectory = slice_mujoco_trajectory_time_window(
                    trajectory,
                    start_step=0,
                    window_steps=int(args.active_window_steps),
                )
            selected.append(trajectory)
    if not selected:
        raise ValueError("Could not select any trajectories for active contact mask construction.")
    return selected


def compute_active_indices_for_training(
    *,
    dataset: ObjectPhysicsDataset,
    object_ids: tuple[str, ...],
    diff_scene,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    trajectories = select_trajectories_for_active_mask(
        dataset=dataset,
        object_ids=object_ids,
        args=args,
        rng=rng,
    )
    active_mask = np.zeros(len(diff_scene.local_surface_points_np), dtype=bool)
    for trajectory_idx, trajectory in enumerate(trajectories, start=1):
        indices = compute_active_contact_point_indices(
            local_surface_points=diff_scene.local_surface_points_np,
            trajectory=trajectory,
            floor_top_z=diff_scene.floor_top_z,
            contact_threshold=float(args.contact_mask_threshold),
        )
        active_mask[indices] = True
        if trajectory_idx == 1 or trajectory_idx == len(trajectories) or trajectory_idx % 64 == 0:
            log(f"active-mask progress {trajectory_idx}/{len(trajectories)} trajectories")
    active_indices = np.flatnonzero(active_mask).astype(np.int32)
    if len(active_indices) == 0:
        raise RuntimeError(
            "No active contact points were detected. Increase --contact-mask-threshold or check surface sampling."
        )
    return active_indices


def build_model(
    *,
    args: argparse.Namespace,
    point_feature_dim: int,
    visual_feature_dim: int,
    torch_device: torch.device,
) -> TrajectoryConditionedFrictionModel:
    model = TrajectoryConditionedFrictionModel.from_dimensions(
        point_feature_dim=int(point_feature_dim),
        encoder_feature_dim=len(ENCODER_FEATURE_SCHEMA),
        latent_dim=int(args.latent_dim),
        projection_dim=int(args.projection_dim),
        step_hidden_dim=int(args.step_hidden_dim),
        gru_hidden_dim=int(args.gru_hidden_dim),
        trajectory_embedding_dim=int(args.trajectory_embedding_dim),
        set_hidden_dim=int(args.set_hidden_dim),
        visual_feature_dim=int(visual_feature_dim),
        visual_hidden_dim=int(args.visual_hidden_dim),
        visual_embedding_dim=int(args.visual_embedding_dim),
        visual_point_hidden_layers=int(args.visual_point_hidden_layers),
        decoder_hidden_dim=int(args.decoder_hidden_dim),
        decoder_hidden_layers=int(args.decoder_hidden_layers),
        decoder_conditioning=str(args.decoder_conditioning),
        decoder_activation=str(args.decoder_activation),
        decoder_basis_count=int(args.decoder_basis_count),
        decoder_basis_base_mode=str(args.decoder_basis_base_mode),
        decoder_basis_normalization=str(args.decoder_basis_normalization),
        decoder_basis_activation=str(args.decoder_basis_activation),
        decoder_basis_norm_eps=float(args.decoder_basis_norm_eps),
        decoder_latent_normalization=str(args.decoder_latent_normalization),
        decoder_raw_limit=None if args.decoder_raw_limit is None else float(args.decoder_raw_limit),
        mu_min=float(args.min_point_friction),
        mu_max=float(args.max_point_friction),
        initial_mu=float(args.point_friction),
    )
    return model.to(torch_device)


@wp.kernel
def apply_batched_external_and_batched_surface_point_forces_trajectory_kernel(
    step_idx: int,
    box_body_ids: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    local_surface_points: wp.array(dtype=wp.vec3),
    weighted_masses: wp.array(dtype=float),
    total_weighted_mass: wp.array(dtype=float),
    batched_point_friction: wp.array(dtype=float),
    step_forces: wp.array(dtype=wp.vec3),
    force_point_offsets_local: wp.array(dtype=wp.vec3),
    trajectory_step_counts: wp.array(dtype=wp.int32),
    batch_size: int,
    point_count: int,
    max_steps: int,
    total_mass: float,
    gravity_magnitude: float,
    floor_top_z: float,
    contact_stiffness: float,
    contact_damping: float,
    contact_band: float,
    friction_regularization: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    batch_idx = tid // point_count
    point_idx = tid - batch_idx * point_count
    if step_idx >= trajectory_step_counts[batch_idx]:
        return

    body_id = box_body_ids[batch_idx]
    pose = body_q[body_id]
    qd = body_qd[body_id]
    world_com = wp.transform_point(pose, body_com[body_id])
    step_offset = batch_idx * max_steps + step_idx

    if point_idx == 0:
        external_force = step_forces[step_offset]
        application_point = wp.transform_point(pose, force_point_offsets_local[batch_idx])
        external_moment_arm = application_point - world_com
        external_torque = wp.cross(external_moment_arm, external_force)
        wp.atomic_add(body_f, body_id, wp.spatial_vector(external_force, external_torque))

    weighted_mass_idx = step_idx * batch_size * point_count + tid
    total_weight_idx = step_idx * batch_size + batch_idx
    total_weight = total_weighted_mass[total_weight_idx]
    if total_weight <= 1.0e-8:
        return

    world_point = wp.transform_point(pose, local_surface_points[point_idx])
    moment_arm = world_point - world_com

    linear_velocity = wp.spatial_top(qd)
    angular_velocity = wp.spatial_bottom(qd)
    point_velocity = linear_velocity + wp.cross(angular_velocity, moment_arm)

    gap = world_point[2] - floor_top_z
    penetration = wp.max(-gap, 0.0)
    safe_band = wp.max(contact_band, 1.0e-6)
    activation = _smoothstep01((contact_band - gap) / safe_band)
    mass_fraction = weighted_masses[weighted_mass_idx] / total_weight

    external_force = step_forces[step_offset]
    normal_load_total = wp.max(0.0, total_mass * gravity_magnitude - external_force[2])
    support_force_z = mass_fraction * normal_load_total
    penalty_force_z = mass_fraction * activation * (
        contact_stiffness * penetration + contact_damping * wp.max(-point_velocity[2], 0.0)
    )
    normal_force = wp.vec3(0.0, 0.0, support_force_z + penalty_force_z)
    tangential_velocity = wp.vec3(point_velocity[0], point_velocity[1], 0.0)
    tangential_speed = wp.sqrt(
        wp.dot(tangential_velocity, tangential_velocity) + friction_regularization * friction_regularization
    )
    normal_load = mass_fraction * normal_load_total
    mu = wp.max(batched_point_friction[batch_idx * point_count + point_idx], 0.0)
    friction_force = -mu * normal_load * (tangential_velocity / tangential_speed)
    total_force = normal_force + friction_force
    total_torque = wp.cross(moment_arm, total_force)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(total_force, total_torque))


def forward_rollout_with_batched_friction_trajectory_loss(
    *,
    diff_scene,
    buffers,
    batched_point_friction: wp.array,
    args: argparse.Namespace,
) -> wp.array:
    point_count = len(diff_scene.local_surface_points_np)
    point_scale = resolve_point_position_loss_scale(args, point_count)

    buffers.loss.zero_()
    buffers.position_loss.zero_()
    buffers.orientation_loss.zero_()
    buffers.linear_velocity_loss.zero_()
    buffers.angular_velocity_loss.zero_()
    buffers.batch_loss.zero_()
    buffers.contact_weighted_masses.zero_()
    buffers.contact_weighted_mass_total.zero_()

    wp.launch(
        set_batched_box_initial_states_kernel,
        dim=buffers.batch_size,
        inputs=[
            diff_scene.box_body_ids_wp,
            buffers.initial_positions,
            buffers.initial_quaternions,
            buffers.initial_linear_velocity,
            buffers.initial_angular_velocity,
            diff_scene.states[0].body_q,
            diff_scene.states[0].body_qd,
        ],
        device=diff_scene.model.device,
    )

    wp.launch(
        accumulate_batched_frame_loss_kernel,
        dim=buffers.batch_size * point_count,
        inputs=[
            0,
            diff_scene.box_body_ids_wp,
            diff_scene.states[0].body_q,
            diff_scene.states[0].body_qd,
            diff_scene.local_surface_points_wp,
            buffers.target_positions,
            buffers.target_quaternions,
            buffers.target_linear_velocity,
            buffers.target_angular_velocity,
            buffers.trajectory_step_counts,
            buffers.frame_scales,
            float(point_scale),
            point_count,
            buffers.max_frames,
            buffers.position_loss,
            buffers.orientation_loss,
            buffers.linear_velocity_loss,
            buffers.angular_velocity_loss,
        ],
        device=diff_scene.model.device,
    )

    for step_idx in range(buffers.max_steps):
        state_in = diff_scene.states[step_idx]
        state_out = diff_scene.states[step_idx + 1]
        state_in.clear_forces()

        wp.launch(
            compute_batched_contact_weighted_masses_kernel,
            dim=buffers.batch_size * point_count,
            inputs=[
                step_idx,
                diff_scene.box_body_ids_wp,
                state_in.body_q,
                diff_scene.local_surface_points_wp,
                diff_scene.point_masses_wp,
                buffers.batch_size,
                point_count,
                float(diff_scene.floor_top_z),
                float(args.friction_contact_threshold),
                buffers.contact_weighted_masses,
                buffers.contact_weighted_mass_total,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            apply_batched_external_and_batched_surface_point_forces_trajectory_kernel,
            dim=buffers.batch_size * point_count,
            inputs=[
                step_idx,
                diff_scene.box_body_ids_wp,
                state_in.body_q,
                state_in.body_qd,
                diff_scene.model.body_com,
                diff_scene.local_surface_points_wp,
                buffers.contact_weighted_masses,
                buffers.contact_weighted_mass_total,
                batched_point_friction,
                buffers.step_forces,
                buffers.force_point_offsets_local,
                buffers.trajectory_step_counts,
                buffers.batch_size,
                point_count,
                buffers.max_steps,
                float(diff_scene.box_mass),
                float(GRAVITY_MAGNITUDE),
                float(diff_scene.floor_top_z),
                float(args.contact_stiffness),
                float(args.contact_damping),
                float(args.friction_contact_threshold),
                float(args.friction_regularization),
                state_in.body_f,
            ],
            device=diff_scene.model.device,
        )

        diff_scene.collision_pipeline.collide(state_in, diff_scene.contacts)
        diff_scene.solver.step(state_in, state_out, diff_scene.control, diff_scene.contacts, float(args.dt))

        wp.launch(
            accumulate_batched_frame_loss_kernel,
            dim=buffers.batch_size * point_count,
            inputs=[
                step_idx + 1,
                diff_scene.box_body_ids_wp,
                state_out.body_q,
                state_out.body_qd,
                diff_scene.local_surface_points_wp,
                buffers.target_positions,
                buffers.target_quaternions,
                buffers.target_linear_velocity,
                buffers.target_angular_velocity,
                buffers.trajectory_step_counts,
                buffers.frame_scales,
                float(point_scale),
                point_count,
                buffers.max_frames,
                buffers.position_loss,
                buffers.orientation_loss,
                buffers.linear_velocity_loss,
                buffers.angular_velocity_loss,
            ],
            device=diff_scene.model.device,
        )

    wp.launch(
        combine_batched_loss_components_kernel,
        dim=buffers.batch_size,
        inputs=[
            buffers.position_loss,
            buffers.orientation_loss,
            buffers.linear_velocity_loss,
            buffers.angular_velocity_loss,
            float(args.position_loss_weight),
            float(args.orientation_loss_weight),
            float(args.linear_velocity_loss_weight),
            float(args.angular_velocity_loss_weight),
            buffers.loss,
        ],
        device=diff_scene.model.device,
    )
    wp.launch(
        sum_batched_losses_kernel,
        dim=buffers.batch_size,
        inputs=[buffers.loss, float(1.0 / max(buffers.batch_size, 1)), buffers.batch_loss],
        device=diff_scene.model.device,
    )
    return buffers.batch_loss


def rollout_view_and_backpropagate(
    *,
    model: TrajectoryConditionedFrictionModel,
    point_features: torch.Tensor,
    latent: torch.Tensor,
    trajectories: tuple[MujocoTrajectory, ...],
    diff_scene,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    active_indices: np.ndarray,
    active_indices_torch: torch.Tensor,
    args: argparse.Namespace,
    view_scale: float,
) -> RolloutDiagnostics:
    active_mu = model.decode_friction(
        point_features,
        latent.reshape(1, -1),
        active_indices=active_indices_torch,
    ).reshape(-1)

    buffers = build_batched_optimization_buffers(
        diff_scene,
        list(trajectories),
        args,
        active_indices,
    )
    buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
    active_mu_warp_source = active_mu.detach().contiguous()
    buffers.active_point_friction = wp.from_torch(
        active_mu_warp_source,
        dtype=wp.float32,
        requires_grad=True,
    )
    clear_batched_optimization_grads(buffers)

    tape = wp.Tape()
    with tape:
        reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
        forward_rollout_with_batched_trajectory_loss(
            diff_scene,
            buffers,
            args,
            scatter_active_point_friction_kernel=scatter_active_point_friction_kernel,
            compute_batched_contact_weighted_masses_kernel=compute_batched_contact_weighted_masses_kernel,
            apply_batched_external_and_surface_point_forces_trajectory_kernel=(
                apply_batched_external_and_surface_point_forces_trajectory_kernel
            ),
            accumulate_batched_frame_loss_kernel=accumulate_batched_frame_loss_kernel,
            combine_batched_loss_components_kernel=combine_batched_loss_components_kernel,
            sum_batched_losses_kernel=sum_batched_losses_kernel,
        )
    tape.backward(buffers.batch_loss)

    if buffers.active_point_friction.grad is None:
        tape.zero()
        raise RuntimeError("Warp rollout did not produce gradients for active_point_friction.")

    grad_unscaled = wp.to_torch(buffers.active_point_friction.grad).to(dtype=active_mu.dtype)
    if not bool(torch.isfinite(grad_unscaled).all().detach().cpu().item()):
        tape.zero()
        raise FloatingPointError("active_point_friction gradient contains non-finite values.")

    active_mu.backward(grad_unscaled * float(view_scale), retain_graph=True)

    losses_t = wp.to_torch(buffers.loss).to(dtype=torch.float64)
    position_t = wp.to_torch(buffers.position_loss).to(dtype=torch.float64)
    orientation_t = wp.to_torch(buffers.orientation_loss).to(dtype=torch.float64)
    linear_t = wp.to_torch(buffers.linear_velocity_loss).to(dtype=torch.float64)
    angular_t = wp.to_torch(buffers.angular_velocity_loss).to(dtype=torch.float64)
    mu_t = active_mu.detach().to(dtype=torch.float64)
    grad_stats_t = grad_unscaled.detach().to(dtype=torch.float64)
    diagnostics = RolloutDiagnostics(
        loss=float(losses_t.mean().detach().cpu().item()) if losses_t.numel() else float("nan"),
        position_loss=float(position_t.mean().detach().cpu().item()) if position_t.numel() else float("nan"),
        orientation_loss=float(orientation_t.mean().detach().cpu().item()) if orientation_t.numel() else float("nan"),
        linear_velocity_loss=float(linear_t.mean().detach().cpu().item()) if linear_t.numel() else float("nan"),
        angular_velocity_loss=float(angular_t.mean().detach().cpu().item()) if angular_t.numel() else float("nan"),
        grad_norm=float(torch.linalg.norm(grad_stats_t).detach().cpu().item()) if grad_stats_t.numel() else 0.0,
        grad_abs_mean=float(grad_stats_t.abs().mean().detach().cpu().item()) if grad_stats_t.numel() else 0.0,
        grad_abs_max=float(grad_stats_t.abs().max().detach().cpu().item()) if grad_stats_t.numel() else 0.0,
        mu_mean=float(mu_t.mean().detach().cpu().item()) if mu_t.numel() else float("nan"),
        mu_std=float(mu_t.std(unbiased=False).detach().cpu().item()) if mu_t.numel() else float("nan"),
        mu_min=float(mu_t.min().detach().cpu().item()) if mu_t.numel() else float("nan"),
        mu_max=float(mu_t.max().detach().cpu().item()) if mu_t.numel() else float("nan"),
    )
    tape.zero()
    return diagnostics


def rollout_diagnostics_to_dict(diagnostics: RolloutDiagnostics) -> dict[str, float]:
    return {
        key: float(getattr(diagnostics, key))
        for key in RolloutDiagnostics.__dataclass_fields__
    }


def decoder_diagnostics_to_dict(
    *,
    model: TrajectoryConditionedFrictionModel,
    point_features: torch.Tensor,
    latent: torch.Tensor,
    active_indices_torch: torch.Tensor,
) -> dict[str, float | str]:
    if getattr(model.friction_decoder, "conditioning", None) != "basis":
        return {}
    return model.friction_decoder.basis_diagnostics(
        point_features,
        latent.reshape(1, -1),
        active_indices=active_indices_torch,
    )


def mean_numeric_subrecords(records: list[dict], field: str) -> dict[str, float | str]:
    numeric_values: dict[str, list[float]] = {}
    string_values: dict[str, str] = {}
    for record in records:
        values = record.get(field, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, str):
                string_values.setdefault(str(key), value)
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            numeric_values.setdefault(str(key), []).append(numeric)
    result: dict[str, float | str] = dict(string_values)
    for key, values in numeric_values.items():
        if values:
            result[key] = float(np.mean(values))
    return result


def _mean_tensor_scalar(values: torch.Tensor) -> float:
    return float(values.mean().detach().cpu().item()) if values.numel() else float("nan")


def _std_tensor_scalar(values: torch.Tensor) -> float:
    return float(values.std(unbiased=False).detach().cpu().item()) if values.numel() else float("nan")


def _min_tensor_scalar(values: torch.Tensor) -> float:
    return float(values.min().detach().cpu().item()) if values.numel() else float("nan")


def _max_tensor_scalar(values: torch.Tensor) -> float:
    return float(values.max().detach().cpu().item()) if values.numel() else float("nan")


def rollout_all_views_and_backpropagate(
    *,
    model: TrajectoryConditionedFrictionModel,
    feature_cache: PointFeatureCache,
    samples,
    latent_output_a,
    latent_output_b,
    diff_scene,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    active_indices: np.ndarray,
    active_indices_torch: torch.Tensor,
    args: argparse.Namespace,
    torch_device: torch.device,
    collect_records: bool = True,
) -> tuple[dict[str, float], list[dict], list[dict]]:
    point_count = int(len(diff_scene.local_surface_points_np))
    combined_trajectories: list[MujocoTrajectory] = []
    view_plans: list[ViewRolloutPlan] = []
    active_mu_by_view: list[torch.Tensor] = []

    total_query_count = 0
    for sample in samples:
        total_query_count += len(sample.query_a.trajectories)
        total_query_count += len(sample.query_b.trajectories)
    if total_query_count <= 0:
        raise ValueError("Cannot run a parallel rollout with no query trajectories.")

    batched_point_friction_t = torch.full(
        (total_query_count, point_count),
        float(args.point_friction),
        dtype=torch.float32,
        device=torch_device,
    )

    cursor = 0
    for object_idx, sample in enumerate(samples):
        entry = feature_cache.get(sample.object_spec)
        for view_name, latent, query_batch in (
            ("a", latent_output_a.latent[object_idx], sample.query_a),
            ("b", latent_output_b.latent[object_idx], sample.query_b),
        ):
            active_mu = model.decode_friction(
                entry.features,
                latent.reshape(1, -1),
                active_indices=active_indices_torch,
            ).reshape(-1)
            trajectories = tuple(query_batch.trajectories)
            start = cursor
            end = start + len(trajectories)
            if end <= start:
                raise ValueError(f"{sample.object_spec.object_id} view {view_name} has no query trajectories.")
            combined_trajectories.extend(trajectories)
            batched_point_friction_t[start:end, active_indices_torch] = (
                active_mu.detach().reshape(1, -1).expand(end - start, -1)
            )
            active_mu_by_view.append(active_mu)
            view_plans.append(
                ViewRolloutPlan(
                    object_index=object_idx,
                    object_id=sample.object_spec.object_id,
                    view_name=view_name,
                    start=start,
                    end=end,
                    latent=latent,
                    active_mu=active_mu,
                )
            )
            cursor = end

    if cursor != total_query_count:
        raise RuntimeError(f"Internal query count mismatch: cursor={cursor}, total={total_query_count}")

    buffers = build_batched_optimization_buffers(
        diff_scene,
        combined_trajectories,
        args,
        active_indices,
    )
    buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
    batched_point_friction_flat = batched_point_friction_t.reshape(-1).contiguous()
    batched_point_friction_wp = wp.from_torch(
        batched_point_friction_flat,
        dtype=wp.float32,
        requires_grad=True,
    )
    clear_batched_optimization_grads(buffers)

    tape = wp.Tape()
    with tape:
        reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
        forward_rollout_with_batched_friction_trajectory_loss(
            diff_scene=diff_scene,
            buffers=buffers,
            batched_point_friction=batched_point_friction_wp,
            args=args,
        )
    tape.backward(buffers.batch_loss)

    if batched_point_friction_wp.grad is None:
        tape.zero()
        raise RuntimeError("Warp rollout did not produce gradients for batched_point_friction.")
    grad_full = wp.to_torch(batched_point_friction_wp.grad).reshape(total_query_count, point_count)
    if not bool(torch.isfinite(grad_full).all().detach().cpu().item()):
        tape.zero()
        raise FloatingPointError("batched_point_friction gradient contains non-finite values.")

    losses_t = wp.to_torch(buffers.loss).to(dtype=torch.float64)
    position_t = wp.to_torch(buffers.position_loss).to(dtype=torch.float64)
    orientation_t = wp.to_torch(buffers.orientation_loss).to(dtype=torch.float64)
    linear_t = wp.to_torch(buffers.linear_velocity_loss).to(dtype=torch.float64)
    angular_t = wp.to_torch(buffers.angular_velocity_loss).to(dtype=torch.float64)

    view_records: list[dict] = []
    view_diagnostics: list[RolloutDiagnostics] = []
    for plan, active_mu in zip(view_plans, active_mu_by_view, strict=True):
        rows = slice(plan.start, plan.end)
        grad_active = grad_full[rows].index_select(1, active_indices_torch).sum(dim=0).to(dtype=active_mu.dtype)
        active_mu.backward(grad_active, retain_graph=True)

        mu_t = active_mu.detach().to(dtype=torch.float64)
        grad_stats_t = grad_active.detach().to(dtype=torch.float64)
        diagnostics = RolloutDiagnostics(
            loss=_mean_tensor_scalar(losses_t[rows]),
            position_loss=_mean_tensor_scalar(position_t[rows]),
            orientation_loss=_mean_tensor_scalar(orientation_t[rows]),
            linear_velocity_loss=_mean_tensor_scalar(linear_t[rows]),
            angular_velocity_loss=_mean_tensor_scalar(angular_t[rows]),
            grad_norm=float(torch.linalg.norm(grad_stats_t).detach().cpu().item()) if grad_stats_t.numel() else 0.0,
            grad_abs_mean=float(grad_stats_t.abs().mean().detach().cpu().item()) if grad_stats_t.numel() else 0.0,
            grad_abs_max=float(grad_stats_t.abs().max().detach().cpu().item()) if grad_stats_t.numel() else 0.0,
            mu_mean=_mean_tensor_scalar(mu_t),
            mu_std=_std_tensor_scalar(mu_t),
            mu_min=_min_tensor_scalar(mu_t),
            mu_max=_max_tensor_scalar(mu_t),
        )
        view_diagnostics.append(diagnostics)

        if collect_records:
            sample = samples[plan.object_index]
            entry = feature_cache.get(sample.object_spec)
            query_batch = sample.query_a if plan.view_name == "a" else sample.query_b
            friction_spec = sample.object_spec.friction_spec
            latent_np = plan.latent.detach().cpu().numpy().astype(np.float64)
            decoder_diagnostics = decoder_diagnostics_to_dict(
                model=model,
                point_features=entry.features,
                latent=plan.latent,
                active_indices_torch=active_indices_torch,
            )
            view_records.append(
                {
                    "object_index": int(plan.object_index),
                    "object_id": str(plan.object_id),
                    "physical_config_id": str(sample.object_spec.physical_config_id),
                    "partition_family": friction_spec.get("partition_family"),
                    "view": str(plan.view_name),
                    "trajectory_count": int(plan.end - plan.start),
                    "episode_indices": [int(value) for value in query_batch.episode_indices.tolist()],
                    "window_start_min": int(np.min(query_batch.window_start_steps))
                    if len(query_batch.window_start_steps)
                    else 0,
                    "window_start_max": int(np.max(query_batch.window_start_steps))
                    if len(query_batch.window_start_steps)
                    else 0,
                    "window_start_mean": float(np.mean(query_batch.window_start_steps))
                    if len(query_batch.window_start_steps)
                    else 0.0,
                    "friction_spec": friction_spec,
                    "latent": {
                        "norm": float(np.linalg.norm(latent_np)),
                        "mean": float(np.mean(latent_np)),
                        "std": float(np.std(latent_np)),
                        "min": float(np.min(latent_np)),
                        "max": float(np.max(latent_np)),
                        "vector": latent_np.tolist(),
                    },
                    "rollout": rollout_diagnostics_to_dict(diagnostics),
                    "decoder": decoder_diagnostics,
                }
            )

    rollout_stats = mean_diagnostics(view_diagnostics)
    object_records: list[dict] = []
    for object_idx, sample in enumerate(samples):
        records = [record for record in view_records if int(record["object_index"]) == object_idx]
        if not records:
            continue
        rollout_keys = records[0]["rollout"].keys()
        rollout_mean = {
            key: float(np.mean([float(record["rollout"][key]) for record in records]))
            for key in rollout_keys
        }
        decoder_mean = mean_numeric_subrecords(records, "decoder")
        latent_vectors = [
            np.asarray(record.get("latent", {}).get("vector", []), dtype=np.float64)
            for record in records
        ]
        latent_vectors = [values for values in latent_vectors if values.size > 0]
        if latent_vectors:
            latent_stack = np.stack(latent_vectors, axis=0)
            latent_mean = np.mean(latent_stack, axis=0)
            if len(latent_vectors) >= 2:
                same_view_distance = float(np.linalg.norm(latent_vectors[0] - latent_vectors[1]))
            else:
                same_view_distance = 0.0
            latent_summary = {
                "same_view_distance": same_view_distance,
                "norm_mean": float(np.mean(np.linalg.norm(latent_stack, axis=1))),
                "norm_std": float(np.std(np.linalg.norm(latent_stack, axis=1))),
                "mean_vector": latent_mean.tolist(),
            }
        else:
            latent_summary = {
                "same_view_distance": 0.0,
                "norm_mean": 0.0,
                "norm_std": 0.0,
                "mean_vector": [],
            }
        object_records.append(
            {
                "object_index": int(object_idx),
                "object_id": str(sample.object_spec.object_id),
                "physical_config_id": str(sample.object_spec.physical_config_id),
                "partition_family": sample.object_spec.friction_spec.get("partition_family"),
                "views": [str(record["view"]) for record in records],
                "trajectory_count": int(sum(int(record["trajectory_count"]) for record in records)),
                "friction_spec": sample.object_spec.friction_spec,
                "latent": latent_summary,
                "rollout": rollout_mean,
                "decoder": decoder_mean,
            }
        )

    tape.zero()
    return rollout_stats, view_records, object_records


def _stable_sigmoid_scalar(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _stable_softplus_scalar(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def build_object_records_from_views(samples, view_records: list[dict]) -> list[dict]:
    object_records: list[dict] = []
    for object_idx, sample in enumerate(samples):
        records = [record for record in view_records if int(record["object_index"]) == object_idx]
        if not records:
            continue
        rollout_keys = records[0]["rollout"].keys()
        rollout_mean = {
            key: float(np.mean([float(record["rollout"][key]) for record in records]))
            for key in rollout_keys
        }
        decoder_mean = mean_numeric_subrecords(records, "decoder")
        latent_vectors = [
            np.asarray(record.get("latent", {}).get("vector", []), dtype=np.float64)
            for record in records
        ]
        latent_vectors = [values for values in latent_vectors if values.size > 0]
        if latent_vectors:
            latent_stack = np.stack(latent_vectors, axis=0)
            latent_mean = np.mean(latent_stack, axis=0)
            same_view_distance = (
                float(np.linalg.norm(latent_vectors[0] - latent_vectors[1]))
                if len(latent_vectors) >= 2
                else 0.0
            )
            latent_summary = {
                "same_view_distance": same_view_distance,
                "norm_mean": float(np.mean(np.linalg.norm(latent_stack, axis=1))),
                "norm_std": float(np.std(np.linalg.norm(latent_stack, axis=1))),
                "mean_vector": latent_mean.tolist(),
            }
        else:
            latent_summary = {
                "same_view_distance": 0.0,
                "norm_mean": 0.0,
                "norm_std": 0.0,
                "mean_vector": [],
            }
        object_records.append(
            {
                "object_index": int(object_idx),
                "object_id": str(sample.object_spec.object_id),
                "physical_config_id": str(sample.object_spec.physical_config_id),
                "partition_family": sample.object_spec.friction_spec.get("partition_family"),
                "views": [str(record["view"]) for record in records],
                "trajectory_count": int(sum(int(record["trajectory_count"]) for record in records)),
                "friction_spec": sample.object_spec.friction_spec,
                "latent": latent_summary,
                "rollout": rollout_mean,
                "decoder": decoder_mean,
            }
        )
    return object_records


def rollout_all_views_with_reused_swap_and_backpropagate(
    *,
    model: TrajectoryConditionedFrictionModel,
    feature_cache: PointFeatureCache,
    samples,
    latent_output_a,
    latent_output_b,
    diff_scene,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    active_indices: np.ndarray,
    active_indices_torch: torch.Tensor,
    args: argparse.Namespace,
    torch_device: torch.device,
    collect_records: bool = True,
) -> tuple[dict[str, float], list[dict], list[dict], dict[str, float]]:
    point_count = int(len(diff_scene.local_surface_points_np))
    combined_trajectories: list[MujocoTrajectory] = []
    view_plans: list[ViewRolloutPlan] = []
    active_mu_by_view: list[torch.Tensor] = []
    swap_plans: list[SwapRolloutPlan] = []

    swap_enabled = float(args.swap_loss_weight) > 0.0
    requested_swap_count = int(args.swap_query_trajectories_per_view)
    main_query_count = 0
    negative_query_count = 0
    for sample in samples:
        main_query_count += len(sample.query_a.trajectories)
        main_query_count += len(sample.query_b.trajectories)
        if swap_enabled:
            negative_query_count += min(requested_swap_count, len(sample.query_a.trajectories))
            negative_query_count += min(requested_swap_count, len(sample.query_b.trajectories))
    total_query_count = main_query_count + negative_query_count
    if main_query_count <= 0:
        raise ValueError("Cannot run a fused rollout with no main query trajectories.")

    batched_point_friction_t = torch.full(
        (total_query_count, point_count),
        float(args.point_friction),
        dtype=torch.float32,
        device=torch_device,
    )

    cursor = 0
    object_count = len(samples)
    for object_idx, sample in enumerate(samples):
        entry = feature_cache.get(sample.object_spec)
        decoded_mu: dict[tuple[str, str], torch.Tensor] = {}
        decode_latents: list[torch.Tensor] = []
        decode_keys: list[tuple[str, str]] = []
        negative_object_idx = (object_idx + 1) % object_count
        for view_name, latent_output in (("a", latent_output_a), ("b", latent_output_b)):
            decode_keys.append(("main", view_name))
            decode_latents.append(latent_output.latent[object_idx])
            if swap_enabled:
                decode_keys.append(("negative", view_name))
                decode_latents.append(latent_output.latent[negative_object_idx])
        decoded = model.decode_friction(
            entry.features,
            torch.stack(decode_latents, dim=0),
            active_indices=active_indices_torch,
        )
        for key, value in zip(decode_keys, decoded, strict=True):
            decoded_mu[key] = value.reshape(-1)

        for view_name, latent_output in (("a", latent_output_a), ("b", latent_output_b)):
            latent = latent_output.latent[object_idx]
            query_batch = sample.query_a if view_name == "a" else sample.query_b
            trajectories = tuple(query_batch.trajectories)
            start = cursor
            end = start + len(trajectories)
            if end <= start:
                raise ValueError(f"{sample.object_spec.object_id} view {view_name} has no query trajectories.")
            active_mu = decoded_mu[("main", view_name)]
            combined_trajectories.extend(trajectories)
            batched_point_friction_t[start:end, active_indices_torch] = (
                active_mu.detach().reshape(1, -1).expand(end - start, -1)
            )
            active_mu_by_view.append(active_mu)
            view_plans.append(
                ViewRolloutPlan(
                    object_index=object_idx,
                    object_id=sample.object_spec.object_id,
                    view_name=view_name,
                    start=start,
                    end=end,
                    latent=latent,
                    active_mu=active_mu,
                )
            )
            cursor = end

            if swap_enabled:
                negative_sample = samples[negative_object_idx]
                swap_trajectories = trajectories[:requested_swap_count]
                if not swap_trajectories:
                    raise ValueError(f"{sample.object_spec.object_id} view {view_name} has no swap query trajectories.")
                negative_mu = decoded_mu[("negative", view_name)]
                negative_start = cursor
                negative_end = negative_start + len(swap_trajectories)
                combined_trajectories.extend(swap_trajectories)
                batched_point_friction_t[negative_start:negative_end, active_indices_torch] = (
                    negative_mu.detach().reshape(1, -1).expand(negative_end - negative_start, -1)
                )
                cursor = negative_end
                swap_plans.append(
                    SwapRolloutPlan(
                        object_index=object_idx,
                        object_id=str(sample.object_spec.object_id),
                        negative_object_id=str(negative_sample.object_spec.object_id),
                        view_name=str(view_name),
                        positive_start=start,
                        positive_end=start + len(swap_trajectories),
                        negative_start=negative_start,
                        negative_end=negative_end,
                        positive_mu=active_mu,
                        negative_mu=negative_mu,
                    )
                )

    if cursor != total_query_count:
        raise RuntimeError(f"Internal fused query count mismatch: cursor={cursor}, total={total_query_count}")

    buffers = build_batched_optimization_buffers(
        diff_scene,
        combined_trajectories,
        args,
        active_indices,
    )
    buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
    batched_point_friction_wp = wp.from_torch(
        batched_point_friction_t.reshape(-1).contiguous(),
        dtype=wp.float32,
        requires_grad=True,
    )
    clear_batched_optimization_grads(buffers)

    tape = wp.Tape()
    with tape:
        reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
        forward_rollout_with_batched_friction_trajectory_loss(
            diff_scene=diff_scene,
            buffers=buffers,
            batched_point_friction=batched_point_friction_wp,
            args=args,
        )
    tape.backward(buffers.batch_loss)

    if batched_point_friction_wp.grad is None:
        tape.zero()
        raise RuntimeError("Fused rollout did not produce gradients for batched_point_friction.")
    grad_full = wp.to_torch(batched_point_friction_wp.grad).reshape(total_query_count, point_count)
    if not bool(torch.isfinite(grad_full).all().detach().cpu().item()):
        tape.zero()
        raise FloatingPointError("Fused batched_point_friction gradient contains non-finite values.")

    losses_t = wp.to_torch(buffers.loss).to(dtype=torch.float64)
    position_t = wp.to_torch(buffers.position_loss).to(dtype=torch.float64)
    orientation_t = wp.to_torch(buffers.orientation_loss).to(dtype=torch.float64)
    linear_t = wp.to_torch(buffers.linear_velocity_loss).to(dtype=torch.float64)
    angular_t = wp.to_torch(buffers.angular_velocity_loss).to(dtype=torch.float64)

    combined_to_main_scale = float(total_query_count) / float(main_query_count)
    view_records: list[dict] = []
    view_diagnostics: list[RolloutDiagnostics] = []
    for plan, active_mu in zip(view_plans, active_mu_by_view, strict=True):
        rows = slice(plan.start, plan.end)
        grad_active = (
            grad_full[rows].index_select(1, active_indices_torch).sum(dim=0) * combined_to_main_scale
        ).to(dtype=active_mu.dtype)
        active_mu.backward(grad_active, retain_graph=True)

        mu_t = active_mu.detach().to(dtype=torch.float64)
        grad_stats_t = grad_active.detach().to(dtype=torch.float64)
        diagnostics = RolloutDiagnostics(
            loss=_mean_tensor_scalar(losses_t[rows]),
            position_loss=_mean_tensor_scalar(position_t[rows]),
            orientation_loss=_mean_tensor_scalar(orientation_t[rows]),
            linear_velocity_loss=_mean_tensor_scalar(linear_t[rows]),
            angular_velocity_loss=_mean_tensor_scalar(angular_t[rows]),
            grad_norm=float(torch.linalg.norm(grad_stats_t).detach().cpu().item()) if grad_stats_t.numel() else 0.0,
            grad_abs_mean=float(grad_stats_t.abs().mean().detach().cpu().item()) if grad_stats_t.numel() else 0.0,
            grad_abs_max=float(grad_stats_t.abs().max().detach().cpu().item()) if grad_stats_t.numel() else 0.0,
            mu_mean=_mean_tensor_scalar(mu_t),
            mu_std=_std_tensor_scalar(mu_t),
            mu_min=_min_tensor_scalar(mu_t),
            mu_max=_max_tensor_scalar(mu_t),
        )
        view_diagnostics.append(diagnostics)

        if collect_records:
            sample = samples[plan.object_index]
            entry = feature_cache.get(sample.object_spec)
            query_batch = sample.query_a if plan.view_name == "a" else sample.query_b
            friction_spec = sample.object_spec.friction_spec
            latent_np = plan.latent.detach().cpu().numpy().astype(np.float64)
            decoder_diagnostics = decoder_diagnostics_to_dict(
                model=model,
                point_features=entry.features,
                latent=plan.latent,
                active_indices_torch=active_indices_torch,
            )
            view_records.append(
                {
                    "object_index": int(plan.object_index),
                    "object_id": str(plan.object_id),
                    "physical_config_id": str(sample.object_spec.physical_config_id),
                    "partition_family": friction_spec.get("partition_family"),
                    "view": str(plan.view_name),
                    "trajectory_count": int(plan.end - plan.start),
                    "episode_indices": [int(value) for value in query_batch.episode_indices.tolist()],
                    "window_start_min": int(np.min(query_batch.window_start_steps))
                    if len(query_batch.window_start_steps)
                    else 0,
                    "window_start_max": int(np.max(query_batch.window_start_steps))
                    if len(query_batch.window_start_steps)
                    else 0,
                    "window_start_mean": float(np.mean(query_batch.window_start_steps))
                    if len(query_batch.window_start_steps)
                    else 0.0,
                    "friction_spec": friction_spec,
                    "latent": {
                        "norm": float(np.linalg.norm(latent_np)),
                        "mean": float(np.mean(latent_np)),
                        "std": float(np.std(latent_np)),
                        "min": float(np.min(latent_np)),
                        "max": float(np.max(latent_np)),
                        "vector": latent_np.tolist(),
                    },
                    "rollout": rollout_diagnostics_to_dict(diagnostics),
                    "decoder": decoder_diagnostics,
                }
            )

    swap_stats: dict[str, float] = {}
    if swap_plans:
        margin = float(args.swap_loss_margin)
        temperature = float(args.swap_loss_temperature)
        swap_weight = float(args.swap_loss_weight)
        plan_scale = swap_weight / float(len(swap_plans))
        positive_losses: list[float] = []
        negative_losses: list[float] = []
        penalties: list[float] = []
        gaps: list[float] = []
        coefficients: list[float] = []
        mu_changes: list[float] = []
        positive_count_total = 0
        negative_count_total = 0
        for plan in swap_plans:
            positive_rows = slice(plan.positive_start, plan.positive_end)
            negative_rows = slice(plan.negative_start, plan.negative_end)
            positive_count = plan.positive_end - plan.positive_start
            negative_count = plan.negative_end - plan.negative_start
            positive_count_total += positive_count
            negative_count_total += negative_count
            positive_loss = _mean_tensor_scalar(losses_t[positive_rows])
            negative_loss = _mean_tensor_scalar(losses_t[negative_rows])
            argument = (margin + positive_loss - negative_loss) / temperature
            coefficient = _stable_sigmoid_scalar(argument)
            penalty = temperature * _stable_softplus_scalar(argument)
            positive_grad = (
                grad_full[positive_rows].index_select(1, active_indices_torch).sum(dim=0)
                * (float(total_query_count) / float(positive_count))
            ).to(dtype=plan.positive_mu.dtype)
            negative_grad = (
                grad_full[negative_rows].index_select(1, active_indices_torch).sum(dim=0)
                * (float(total_query_count) / float(negative_count))
            ).to(dtype=plan.negative_mu.dtype)
            plan.positive_mu.backward(
                positive_grad * float(plan_scale * coefficient),
                retain_graph=True,
            )
            plan.negative_mu.backward(
                negative_grad * float(-plan_scale * coefficient),
                retain_graph=True,
            )

            positive_losses.append(positive_loss)
            negative_losses.append(negative_loss)
            penalties.append(penalty)
            gaps.append(negative_loss - positive_loss)
            coefficients.append(coefficient)
            mu_changes.append(
                float(torch.mean(torch.abs(plan.positive_mu.detach() - plan.negative_mu.detach())).cpu())
            )
        gap_array = np.asarray(gaps, dtype=np.float64)
        swap_stats = {
            "loss": float(np.mean(penalties)),
            "weighted_loss": float(swap_weight * np.mean(penalties)),
            "positive_rollout_loss": float(np.mean(positive_losses)),
            "swapped_rollout_loss": float(np.mean(negative_losses)),
            "loss_gap": float(np.mean(gap_array)),
            "accuracy": float(np.mean(gap_array > 0.0)),
            "margin_accuracy": float(np.mean(gap_array >= margin)),
            "ranking_coefficient_mean": float(np.mean(coefficients)),
            "mu_abs_change_mean": float(np.mean(mu_changes)),
            "query_trajectory_count": float(positive_count_total + negative_count_total),
            "reused_positive_trajectory_count": float(positive_count_total),
            "extra_negative_trajectory_count": float(negative_count_total),
            "pair_count": float(len(swap_plans)),
        }

    rollout_stats = mean_diagnostics(view_diagnostics)
    object_records = build_object_records_from_views(samples, view_records) if collect_records else []
    tape.zero()
    return rollout_stats, view_records, object_records, swap_stats


def rollout_cyclic_latent_swap_and_backpropagate(
    *,
    model: TrajectoryConditionedFrictionModel,
    feature_cache: PointFeatureCache,
    samples,
    latent_output_a,
    latent_output_b,
    diff_scene,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    active_indices: np.ndarray,
    active_indices_torch: torch.Tensor,
    args: argparse.Namespace,
    torch_device: torch.device,
) -> dict[str, float]:
    swap_weight = float(args.swap_loss_weight)
    if swap_weight <= 0.0:
        return {}
    if len(samples) < 2:
        raise ValueError("Cyclic latent swap requires at least two objects per optimizer step.")

    point_count = int(len(diff_scene.local_surface_points_np))
    requested_query_count = int(args.swap_query_trajectories_per_view)
    total_query_count = 0
    for sample in samples:
        total_query_count += 2 * min(requested_query_count, len(sample.query_a.trajectories))
        total_query_count += 2 * min(requested_query_count, len(sample.query_b.trajectories))
    if total_query_count <= 0:
        raise ValueError("Cannot run cyclic latent swap with no query trajectories.")

    combined_trajectories: list[MujocoTrajectory] = []
    plans: list[SwapRolloutPlan] = []
    batched_point_friction_t = torch.full(
        (total_query_count, point_count),
        float(args.point_friction),
        dtype=torch.float32,
        device=torch_device,
    )

    cursor = 0
    object_count = len(samples)
    for object_idx, sample in enumerate(samples):
        negative_object_idx = (object_idx + 1) % object_count
        negative_sample = samples[negative_object_idx]
        entry = feature_cache.get(sample.object_spec)
        for view_name, positive_latent, negative_latent, query_batch in (
            (
                "a",
                latent_output_a.latent[object_idx],
                latent_output_a.latent[negative_object_idx],
                sample.query_a,
            ),
            (
                "b",
                latent_output_b.latent[object_idx],
                latent_output_b.latent[negative_object_idx],
                sample.query_b,
            ),
        ):
            trajectories = tuple(query_batch.trajectories[:requested_query_count])
            if not trajectories:
                raise ValueError(f"{sample.object_spec.object_id} view {view_name} has no swap query trajectories.")
            positive_mu = model.decode_friction(
                entry.features,
                positive_latent.reshape(1, -1),
                active_indices=active_indices_torch,
            ).reshape(-1)
            negative_mu = model.decode_friction(
                entry.features,
                negative_latent.reshape(1, -1),
                active_indices=active_indices_torch,
            ).reshape(-1)

            positive_start = cursor
            positive_end = positive_start + len(trajectories)
            combined_trajectories.extend(trajectories)
            batched_point_friction_t[positive_start:positive_end, active_indices_torch] = (
                positive_mu.detach().reshape(1, -1).expand(len(trajectories), -1)
            )
            cursor = positive_end

            negative_start = cursor
            negative_end = negative_start + len(trajectories)
            combined_trajectories.extend(trajectories)
            batched_point_friction_t[negative_start:negative_end, active_indices_torch] = (
                negative_mu.detach().reshape(1, -1).expand(len(trajectories), -1)
            )
            cursor = negative_end

            plans.append(
                SwapRolloutPlan(
                    object_index=object_idx,
                    object_id=str(sample.object_spec.object_id),
                    negative_object_id=str(negative_sample.object_spec.object_id),
                    view_name=view_name,
                    positive_start=positive_start,
                    positive_end=positive_end,
                    negative_start=negative_start,
                    negative_end=negative_end,
                    positive_mu=positive_mu,
                    negative_mu=negative_mu,
                )
            )

    if cursor != total_query_count:
        raise RuntimeError(f"Internal swap query count mismatch: cursor={cursor}, total={total_query_count}")

    buffers = build_batched_optimization_buffers(
        diff_scene,
        combined_trajectories,
        args,
        active_indices,
    )
    buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
    batched_point_friction_wp = wp.from_torch(
        batched_point_friction_t.reshape(-1).contiguous(),
        dtype=wp.float32,
        requires_grad=True,
    )
    clear_batched_optimization_grads(buffers)

    tape = wp.Tape()
    with tape:
        reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
        forward_rollout_with_batched_friction_trajectory_loss(
            diff_scene=diff_scene,
            buffers=buffers,
            batched_point_friction=batched_point_friction_wp,
            args=args,
        )
    tape.backward(buffers.batch_loss)

    if batched_point_friction_wp.grad is None:
        tape.zero()
        raise RuntimeError("Warp swap rollout did not produce gradients for batched_point_friction.")
    grad_full = wp.to_torch(batched_point_friction_wp.grad).reshape(total_query_count, point_count)
    if not bool(torch.isfinite(grad_full).all().detach().cpu().item()):
        tape.zero()
        raise FloatingPointError("Swap batched_point_friction gradient contains non-finite values.")

    losses_t = wp.to_torch(buffers.loss).to(dtype=torch.float64)
    margin = float(args.swap_loss_margin)
    temperature = float(args.swap_loss_temperature)
    plan_scale = swap_weight / float(len(plans))
    positive_losses: list[float] = []
    negative_losses: list[float] = []
    penalties: list[float] = []
    gaps: list[float] = []
    coefficients: list[float] = []
    mu_changes: list[float] = []

    for plan in plans:
        positive_rows = slice(plan.positive_start, plan.positive_end)
        negative_rows = slice(plan.negative_start, plan.negative_end)
        positive_count = plan.positive_end - plan.positive_start
        negative_count = plan.negative_end - plan.negative_start
        positive_loss = _mean_tensor_scalar(losses_t[positive_rows])
        negative_loss = _mean_tensor_scalar(losses_t[negative_rows])
        argument = (margin + positive_loss - negative_loss) / temperature
        coefficient = _stable_sigmoid_scalar(argument)
        penalty = temperature * _stable_softplus_scalar(argument)

        positive_grad = (
            grad_full[positive_rows].index_select(1, active_indices_torch).sum(dim=0)
            * (float(total_query_count) / float(positive_count))
        ).to(dtype=plan.positive_mu.dtype)
        negative_grad = (
            grad_full[negative_rows].index_select(1, active_indices_torch).sum(dim=0)
            * (float(total_query_count) / float(negative_count))
        ).to(dtype=plan.negative_mu.dtype)
        plan.positive_mu.backward(
            positive_grad * float(plan_scale * coefficient),
            retain_graph=True,
        )
        plan.negative_mu.backward(
            negative_grad * float(-plan_scale * coefficient),
            retain_graph=True,
        )

        positive_losses.append(positive_loss)
        negative_losses.append(negative_loss)
        penalties.append(penalty)
        gaps.append(negative_loss - positive_loss)
        coefficients.append(coefficient)
        mu_changes.append(
            float(torch.mean(torch.abs(plan.positive_mu.detach() - plan.negative_mu.detach())).cpu())
        )

    tape.zero()
    gap_array = np.asarray(gaps, dtype=np.float64)
    return {
        "loss": float(np.mean(penalties)),
        "weighted_loss": float(swap_weight * np.mean(penalties)),
        "positive_rollout_loss": float(np.mean(positive_losses)),
        "swapped_rollout_loss": float(np.mean(negative_losses)),
        "loss_gap": float(np.mean(gap_array)),
        "accuracy": float(np.mean(gap_array > 0.0)),
        "margin_accuracy": float(np.mean(gap_array >= margin)),
        "ranking_coefficient_mean": float(np.mean(coefficients)),
        "mu_abs_change_mean": float(np.mean(mu_changes)),
        "query_trajectory_count": float(total_query_count),
        "pair_count": float(len(plans)),
    }


def mean_diagnostics(items: list[RolloutDiagnostics]) -> dict[str, float]:
    if not items:
        return {}
    keys = RolloutDiagnostics.__dataclass_fields__.keys()
    result = {}
    for key in keys:
        values = np.asarray([getattr(item, key) for item in items], dtype=np.float64)
        result[key] = float(np.mean(values))
    return result


def summarize_latents(latent_a: torch.Tensor, latent_b: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        same = torch.linalg.norm(latent_a - latent_b, dim=-1)
        all_latent = torch.cat((latent_a, latent_b), dim=0)
        latent_norm = torch.linalg.norm(all_latent, dim=-1)
        same_mean = float(same.mean().detach().cpu())
        same_max = float(same.max().detach().cpu())
        same_std = float(same.std(unbiased=False).detach().cpu()) if same.numel() else 0.0
        if latent_a.shape[0] >= 2:
            labels = torch.cat(
                (
                    torch.arange(latent_a.shape[0], device=latent_a.device),
                    torch.arange(latent_a.shape[0], device=latent_a.device),
                ),
                dim=0,
            )
            distances = torch.cdist(all_latent, all_latent)
            different = distances[labels[:, None] != labels[None, :]]
            all_pair = torch.pdist(all_latent)
            different_mean = float(different.mean().detach().cpu()) if different.numel() else 0.0
            different_std = float(different.std(unbiased=False).detach().cpu()) if different.numel() else 0.0
            different_min = float(different.min().detach().cpu()) if different.numel() else 0.0
            different_max = float(different.max().detach().cpu()) if different.numel() else 0.0
            all_pair_mean = float(all_pair.mean().detach().cpu()) if all_pair.numel() else 0.0
        else:
            different_mean = 0.0
            different_std = 0.0
            different_min = 0.0
            different_max = 0.0
            all_pair_mean = 0.0
        return {
            "same_object_latent_distance_mean": same_mean,
            "same_object_latent_distance_std": same_std,
            "same_object_latent_distance_max": same_max,
            "different_object_latent_distance_mean": different_mean,
            "different_object_latent_distance_std": different_std,
            "different_object_latent_distance_min": different_min,
            "different_object_latent_distance_max": different_max,
            "same_to_different_latent_distance_ratio": same_mean / max(different_mean, 1.0e-8),
            "all_view_latent_pair_distance_mean": all_pair_mean,
            "latent_norm_mean": float(latent_norm.mean().detach().cpu()),
            "latent_norm_std": float(latent_norm.std(unbiased=False).detach().cpu()) if latent_norm.numel() else 0.0,
        }


RESUME_CRITICAL_ARG_NAMES = (
    "object_split",
    "objects_per_step",
    "object_sampling_strategy",
    "context_trajectories_per_view",
    "query_trajectories_per_view",
    "context_window_steps",
    "query_window_steps",
    "random_context_windows",
    "random_query_windows",
    "time_window_source_max_steps",
    "latent_dim",
    "projection_dim",
    "step_hidden_dim",
    "gru_hidden_dim",
    "trajectory_embedding_dim",
    "set_hidden_dim",
    "visual_hidden_dim",
    "visual_embedding_dim",
    "visual_point_hidden_layers",
    "decoder_hidden_dim",
    "decoder_hidden_layers",
    "decoder_conditioning",
    "decoder_activation",
    "decoder_basis_count",
    "decoder_basis_base_mode",
    "decoder_basis_normalization",
    "decoder_basis_activation",
    "decoder_basis_norm_eps",
    "decoder_latent_normalization",
    "decoder_raw_limit",
    "consistency_weight",
    "contrastive_weight",
    "contrastive_temperature",
    "latent_norm_weight",
    "latent_norm_target",
    "swap_loss_weight",
    "swap_loss_margin",
    "swap_loss_temperature",
    "swap_query_trajectories_per_view",
    "no_dino",
    "decoder_point_feature_mode",
    "dino_to_encoder",
    "dino_neighbor_radius",
    "dino_neighbor_k",
    "dino_position_frequencies",
    "dino_mlp_max_match_distance",
    "dino_feature_normalization",
    "min_point_friction",
    "max_point_friction",
    "point_friction",
    "contact_friction",
    "contact_stiffness",
    "contact_damping",
    "contact_margin",
    "friction_regularization",
    "friction_contact_threshold",
    "contact_mask_threshold",
    "solver_iterations",
    "box_mass",
    "floor_half_extents",
    "box_half_extents",
    "box_start_pos",
    "surface_point_spacing",
    "avoid_zero_surface_point_x",
    "position_loss_weight",
    "orientation_loss_weight",
    "linear_velocity_loss_weight",
    "angular_velocity_loss_weight",
    "point_position_loss_reduction",
    "active_object_limit",
    "active_trajectories_per_object",
    "active_window_steps",
    "active_use_query_pool",
    "learning_rate",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
)


def _resume_value(value):
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_resume_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_resume_checkpoint(
    *,
    payload: dict,
    args: argparse.Namespace,
    object_ids: Sequence[str],
    active_indices: np.ndarray,
    diff_scene,
) -> None:
    saved_args = dict(payload.get("args", {}))
    mismatches = []
    for name in RESUME_CRITICAL_ARG_NAMES:
        if name not in saved_args:
            continue
        current_value = _resume_value(getattr(args, name))
        saved_value = _resume_value(saved_args[name])
        if current_value != saved_value:
            mismatches.append(f"{name}: checkpoint={saved_value!r}, current={current_value!r}")
    if mismatches:
        raise ValueError("Resume checkpoint training configuration mismatch:\n  " + "\n  ".join(mismatches))

    checkpoint_active_indices = np.asarray(payload.get("active_indices"), dtype=np.int32)
    if checkpoint_active_indices.shape != active_indices.shape or not np.array_equal(
        checkpoint_active_indices,
        active_indices,
    ):
        raise ValueError(
            "Resume checkpoint active contact-point indices do not match the current run. "
            "Use the same active-mask and surface-point settings."
        )
    checkpoint_points = np.asarray(payload.get("local_surface_points"), dtype=np.float32)
    current_points = np.asarray(diff_scene.local_surface_points_np, dtype=np.float32)
    if checkpoint_points.shape != current_points.shape or not np.allclose(checkpoint_points, current_points):
        raise ValueError("Resume checkpoint local surface points do not match the current Newton scene.")
    saved_object_ids = payload.get("object_ids")
    if saved_object_ids is not None and tuple(str(value) for value in saved_object_ids) != tuple(object_ids):
        raise ValueError("Resume checkpoint object ids do not match the current object split.")


def reconstruct_appearance_counts(
    *,
    object_ids: Sequence[str],
    history: Sequence[dict],
) -> dict[str, int]:
    counts = {str(object_id): 0 for object_id in object_ids}
    for record in history:
        for object_id in record.get("objects", []):
            object_id = str(object_id)
            if object_id in counts:
                counts[object_id] += 1
    return counts


def infer_best_iteration(history: Sequence[dict], *, fallback: int) -> int:
    finite_records = [
        record
        for record in history
        if math.isfinite(float(record.get("total_loss", float("nan"))))
    ]
    if not finite_records:
        return int(fallback)
    return int(min(finite_records, key=lambda record: float(record["total_loss"]))["iteration"])


def load_resume_checkpoint(
    *,
    path: Path,
    args: argparse.Namespace,
    model: TrajectoryConditionedFrictionModel,
    optimizer: torch.optim.Optimizer,
    object_sampler: ObjectBatchSampler,
    object_ids: Sequence[str],
    active_indices: np.ndarray,
    diff_scene,
    rng: np.random.Generator,
    torch_device: torch.device,
) -> dict:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported resume checkpoint format: {checkpoint_path}")
    validate_resume_checkpoint(
        payload=payload,
        args=args,
        object_ids=object_ids,
        active_indices=active_indices,
        diff_scene=diff_scene,
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    history = list(payload.get("history", []))
    appearance_counts = payload.get("appearance_counts")
    if appearance_counts is None:
        appearance_counts = reconstruct_appearance_counts(object_ids=object_ids, history=history)
        appearance_counts_exact = not bool(payload.get("history_truncated", False))
    else:
        appearance_counts = {
            str(object_id): int(appearance_counts.get(str(object_id), 0))
            for object_id in object_ids
        }
        appearance_counts_exact = True

    if "numpy_rng_state" in payload:
        rng.bit_generator.state = payload["numpy_rng_state"]
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    saved_cuda_states = payload.get("cuda_rng_state_all")
    if saved_cuda_states is not None and torch.cuda.is_available():
        if len(saved_cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Resume checkpoint CUDA RNG state count does not match the current visible CUDA device count."
            )
        torch.cuda.set_rng_state_all([state.cpu() for state in saved_cuda_states])
    if "object_sampler_state" in payload:
        object_sampler.load_state_dict(payload["object_sampler_state"])

    completed_iteration = int(payload["iteration"])
    if int(args.opt_iters) <= completed_iteration:
        raise ValueError(
            f"--opt-iters={int(args.opt_iters)} must be greater than resumed iteration={completed_iteration}."
        )
    return {
        "checkpoint_path": str(checkpoint_path),
        "completed_iteration": completed_iteration,
        "start_iteration": completed_iteration + 1,
        "best_loss": float(payload.get("best_loss", float("inf"))),
        "best_iteration": int(
            payload.get("best_iteration", infer_best_iteration(history, fallback=completed_iteration))
        ),
        "history": history[-100:],
        "appearance_counts": appearance_counts,
        "appearance_counts_exact": bool(appearance_counts_exact),
        "sampler_state_exact": "object_sampler_state" in payload,
        "rng_state_exact": "numpy_rng_state" in payload and "torch_rng_state" in payload,
        "wandb_run_id": payload.get("wandb_run_id"),
    }


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    iteration: int,
    model: TrajectoryConditionedFrictionModel,
    optimizer: torch.optim.Optimizer,
    active_indices: np.ndarray,
    diff_scene,
    feature_metadata: dict | None,
    history: list[dict],
    best_loss: float,
    best_iteration: int,
    object_ids: Sequence[str],
    appearance_counts: dict[str, int],
    object_sampler: ObjectBatchSampler,
    rng: np.random.Generator,
    wandb_run_id: str | None,
) -> dict:
    # The complete logged history is stored in train_history.jsonl. Keeping a
    # short tail here prevents checkpoints from growing linearly with training.
    checkpoint_history = history[-100:]
    return {
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "active_indices": np.asarray(active_indices, dtype=np.int32),
        "local_surface_points": np.asarray(diff_scene.local_surface_points_np, dtype=np.float32),
        "point_masses": np.asarray(diff_scene.point_masses_np, dtype=np.float32),
        "args": _args_dict(args),
        "feature_metadata": _jsonable(feature_metadata or {}),
        "history": checkpoint_history,
        "history_truncated": len(checkpoint_history) < len(history),
        "history_total_logged_records": len(history),
        "best_loss": float(best_loss),
        "best_iteration": int(best_iteration),
        "object_ids": list(object_ids),
        "appearance_counts": {str(key): int(value) for key, value in appearance_counts.items()},
        "object_sampler_state": object_sampler.state_dict(),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "wandb_run_id": wandb_run_id,
        "friction_parameterization": str(args.friction_parameterization),
        "uses_contact_value_field": False,
        "contact_value_names": ["mu"],
    }


def _clone_checkpoint_value_for_deferred_save(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_checkpoint_value_for_deferred_save(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_checkpoint_value_for_deferred_save(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_value_for_deferred_save(item) for item in value)
    return copy.deepcopy(value)


def clone_checkpoint_payload_for_deferred_save(payload: dict) -> dict:
    return _clone_checkpoint_value_for_deferred_save(payload)


def save_checkpoint(
    *,
    path: Path,
    args: argparse.Namespace,
    iteration: int,
    model: TrajectoryConditionedFrictionModel,
    optimizer: torch.optim.Optimizer,
    active_indices: np.ndarray,
    diff_scene,
    feature_metadata: dict | None,
    history: list[dict],
    best_loss: float,
    best_iteration: int,
    object_ids: Sequence[str],
    appearance_counts: dict[str, int],
    object_sampler: ObjectBatchSampler,
    rng: np.random.Generator,
    wandb_run_id: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            args=args,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            active_indices=active_indices,
            diff_scene=diff_scene,
            feature_metadata=feature_metadata,
            history=history,
            best_loss=best_loss,
            best_iteration=best_iteration,
            object_ids=object_ids,
            appearance_counts=appearance_counts,
            object_sampler=object_sampler,
            rng=rng,
            wandb_run_id=wandb_run_id,
        ),
        path,
    )


@torch.no_grad()
def export_preview_point_clouds(
    *,
    model: TrajectoryConditionedFrictionModel,
    feature_cache: PointFeatureCache,
    dataset: ObjectPhysicsDataset,
    object_ids: tuple[str, ...],
    active_indices: np.ndarray,
    diff_scene,
    args: argparse.Namespace,
    output_dir: Path,
    torch_device: torch.device,
) -> None:
    count = max(int(args.export_preview_count), 0)
    if count <= 0:
        return
    preview_dir = output_dir / "preview_point_clouds"
    preview_dir.mkdir(parents=True, exist_ok=True)
    color_min = args.point_cloud_color_min
    color_max = args.point_cloud_color_max
    if color_min is None:
        color_min = float(args.min_point_friction)
    if color_max is None:
        color_max = float(args.max_point_friction)
    for object_id in object_ids[:count]:
        obj = dataset.get_object(object_id)
        entry = feature_cache.get(obj)
        sample = dataset.sample_object_training_data(
            object_id,
            context_trajectories_per_view=int(args.context_trajectories_per_view),
            query_trajectories_per_view=1,
            context_window_steps=int(args.context_window_steps),
            query_window_steps=int(args.query_window_steps),
            random_context_windows=False,
            random_query_windows=False,
            rng=np.random.default_rng(int(args.seed) + 991),
        )
        context, mask = stack_encoder_batches([sample.context_a], device=torch_device)
        visual_features = stack_visual_features([entry], device=torch_device)
        latent = model.encode_context(
            context,
            context_valid_mask=mask,
            visual_features=visual_features,
        ).latent[0]
        all_indices = torch.arange(len(diff_scene.local_surface_points_np), dtype=torch.long, device=torch_device)
        full_mu = model.decode_friction(entry.features, latent.reshape(1, -1), active_indices=all_indices).reshape(-1)
        full_mu_np = full_mu.detach().cpu().numpy().astype(np.float32)
        save_contact_friction_point_cloud(
            local_surface_points=diff_scene.local_surface_points_np,
            point_friction=full_mu_np,
            output_path=preview_dir / f"{object_id}.ply",
            active_indices=active_indices,
            color_min=float(color_min),
            color_max=float(color_max),
        )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")


def init_wandb_run(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    config_payload: dict,
    trainable_params: int,
) -> object | None:
    if not bool(args.wandb):
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "W&B logging was requested, but the 'wandb' package is not installed. "
            "Install wandb or rerun with --no-wandb."
        ) from exc

    run_name = args.wandb_run_name or output_dir.name
    wandb_dir = Path(args.wandb_dir).expanduser().resolve() if args.wandb_dir is not None else output_dir
    wandb_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **_jsonable(config_payload),
        "trainable_params": int(trainable_params),
        "experiment_dir": str(output_dir),
    }
    init_kwargs = dict(
        project=str(args.wandb_project),
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        mode=str(args.wandb_mode),
        dir=str(wandb_dir),
        tags=args.wandb_tags,
        config=config,
        save_code=True,
    )
    if args.wandb_resume_id is not None:
        init_kwargs["id"] = str(args.wandb_resume_id)
        init_kwargs["resume"] = "allow"
    return wandb.init(**init_kwargs)


def flatten_numeric_payload(prefix: str, payload: dict) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_numeric_payload(name, value))
        elif isinstance(value, bool):
            result[name] = int(value)
        elif isinstance(value, (int, np.integer)):
            result[name] = int(value)
        elif isinstance(value, (float, np.floating)):
            value_float = float(value)
            if math.isfinite(value_float):
                result[name] = value_float
        elif isinstance(value, str):
            result[name] = value
    return result


def add_rollout_distribution_metrics(
    payload: dict,
    *,
    prefix: str,
    records: list[dict],
    keys: tuple[str, ...] = ("loss", "mu_mean", "mu_std", "mu_min", "mu_max", "grad_norm"),
) -> None:
    if not records:
        return
    for key in keys:
        values = []
        for record in records:
            rollout = record.get("rollout", {})
            if key not in rollout:
                continue
            value = float(rollout[key])
            if math.isfinite(value):
                values.append(value)
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        payload[f"{prefix}/{key}_mean"] = float(np.mean(arr))
        payload[f"{prefix}/{key}_std"] = float(np.std(arr))
        payload[f"{prefix}/{key}_min"] = float(np.min(arr))
        payload[f"{prefix}/{key}_max"] = float(np.max(arr))


def add_partition_family_counts(payload: dict, object_metrics: list[dict]) -> None:
    counts: dict[str, int] = {}
    for item in object_metrics:
        family = str(item.get("partition_family") or "unknown")
        counts[family] = counts.get(family, 0) + 1
    for family, count in sorted(counts.items()):
        safe_family = "".join(char if char.isalnum() else "_" for char in family).strip("_") or "unknown"
        payload[f"batch/partition_family_count/{safe_family}"] = int(count)


def wandb_metric_component(value: str) -> str:
    result = []
    previous_separator = False
    for char in str(value):
        if char.isalnum():
            result.append(char)
            previous_separator = False
        elif not previous_separator:
            result.append("_")
            previous_separator = True
    return "".join(result).strip("_") or "unnamed"


def add_numeric_mapping_metrics(payload: dict, *, prefix: str, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, (int, np.integer)):
            payload[f"{prefix}/{key}"] = int(value)
        elif isinstance(value, (float, np.floating)):
            value_float = float(value)
            if math.isfinite(value_float):
                payload[f"{prefix}/{key}"] = value_float


def add_latent_metrics(payload: dict, *, prefix: str, latent: dict) -> None:
    for key in ("norm", "mean", "std", "min", "max", "same_view_distance", "norm_mean", "norm_std"):
        if key not in latent:
            continue
        value = float(latent[key])
        if math.isfinite(value):
            payload[f"{prefix}/{key}"] = value

    vector = latent.get("vector")
    if vector is None:
        vector = latent.get("mean_vector")
    if vector is None:
        return
    for dim, value in enumerate(vector):
        value_float = float(value)
        if math.isfinite(value_float):
            payload[f"{prefix}/z_{dim:02d}"] = value_float


def add_rollout_record_metrics(payload: dict, *, prefix: str, rollout: dict) -> None:
    for key in (
        "loss",
        "position_loss",
        "orientation_loss",
        "linear_velocity_loss",
        "angular_velocity_loss",
        "mu_mean",
        "mu_std",
        "mu_min",
        "mu_max",
        "grad_norm",
    ):
        if key not in rollout:
            continue
        value = float(rollout[key])
        if math.isfinite(value):
            payload[f"{prefix}/{key}"] = value


def add_hierarchical_view_and_object_metrics(
    payload: dict,
    *,
    object_metrics: list[dict],
    view_metrics: list[dict],
) -> None:
    for item in object_metrics:
        object_id = str(item.get("object_id", "object"))
        object_key = wandb_metric_component(object_id)
        prefix = f"objects/{object_key}"
        payload[f"{prefix}/present"] = 1
        payload[f"{prefix}/trajectory_count"] = int(item.get("trajectory_count", 0))
        add_rollout_record_metrics(payload, prefix=f"{prefix}/rollout", rollout=item.get("rollout", {}))
        add_latent_metrics(payload, prefix=f"{prefix}/latent", latent=item.get("latent", {}))
        add_numeric_mapping_metrics(payload, prefix=f"{prefix}/sampling", values=item.get("sampling", {}))
        add_numeric_mapping_metrics(
            payload,
            prefix=f"{prefix}/target_friction",
            values=item.get("friction_spec", {}),
        )

    for item in view_metrics:
        object_id = str(item.get("object_id", "object"))
        object_key = wandb_metric_component(object_id)
        view_key = wandb_metric_component(str(item.get("view", "view")))
        prefix = f"views/{object_key}/{view_key}"
        payload[f"{prefix}/present"] = 1
        payload[f"{prefix}/trajectory_count"] = int(item.get("trajectory_count", 0))
        payload[f"{prefix}/window_start_min"] = int(item.get("window_start_min", 0))
        payload[f"{prefix}/window_start_max"] = int(item.get("window_start_max", 0))
        payload[f"{prefix}/window_start_mean"] = float(item.get("window_start_mean", 0.0))
        add_rollout_record_metrics(payload, prefix=f"{prefix}/rollout", rollout=item.get("rollout", {}))
        add_latent_metrics(payload, prefix=f"{prefix}/latent", latent=item.get("latent", {}))
        add_numeric_mapping_metrics(
            payload,
            prefix=f"{prefix}/target_friction",
            values=item.get("friction_spec", {}),
        )


def build_wandb_log_payload(
    record: dict,
    *,
    best_loss: float,
    best_iteration: int,
    detail: str = "full",
) -> dict:
    object_metrics = list(record.get("object_metrics", []))
    view_metrics = list(record.get("view_metrics", []))
    object_ids = [str(item.get("object_id", "")) for item in object_metrics]
    payload = {
        "train/total_loss": float(record["total_loss"]),
        "train/rollout_loss": float(record["rollout_loss"]),
        "train/swap_loss": float(record.get("swap_loss", 0.0)),
        "train/swap_weighted_loss": float(record.get("swap_weighted_loss", 0.0)),
        "train/regularization_total": float(record["regularization_total"]),
        "train/consistency_loss": float(record["consistency_loss"]),
        "train/contrastive_loss": float(record["contrastive_loss"]),
        "train/latent_norm_loss": float(record.get("latent_norm_loss", 0.0)),
        "train/torch_grad_norm_before_clip": float(record["torch_grad_norm_before_clip"]),
        "train/elapsed_sec": float(record["elapsed_sec"]),
        "train/best_loss": float(best_loss),
        "train/best_iteration": int(best_iteration),
        "train/object_count_this_step": int(len(record.get("objects", []))),
        "batch/object_ids": ",".join(object_ids),
        "batch/view_count": int(len(view_metrics)),
        "batch/query_trajectory_count": int(
            sum(int(item.get("trajectory_count", 0)) for item in view_metrics)
        ),
    }
    payload.update(flatten_numeric_payload("rollout", record.get("rollout", {})))
    payload.update(flatten_numeric_payload("swap", record.get("swap", {})))
    payload.update(flatten_numeric_payload("latent", record.get("latent", {})))
    payload.update(flatten_numeric_payload("object_sampling", record.get("object_sampling", {})))
    add_rollout_distribution_metrics(
        payload,
        prefix="objects_rollout",
        records=object_metrics,
        keys=("loss", "mu_mean"),
    )
    add_rollout_distribution_metrics(
        payload,
        prefix="views_rollout",
        records=view_metrics,
        keys=("loss",),
    )
    add_partition_family_counts(payload, object_metrics)
    if str(detail) == "full":
        add_hierarchical_view_and_object_metrics(
            payload,
            object_metrics=object_metrics,
            view_metrics=view_metrics,
        )
    return payload


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.experiment_dir).expanduser().resolve()
    resume_checkpoint_path = (
        None if args.resume_checkpoint is None else Path(args.resume_checkpoint).expanduser().resolve()
    )
    if resume_checkpoint_path is not None and resume_checkpoint_path.parent != output_dir:
        raise ValueError(
            "Full resume must use the original experiment directory so history and best/last checkpoints "
            f"remain together. Got experiment_dir={output_dir} and checkpoint parent={resume_checkpoint_path.parent}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        shutil.copy2(manifest_path, output_dir / "manifest.json")

    log(f"loading object manifest: {manifest_path}")
    dataset = ObjectPhysicsDataset(
        manifest_path,
        cache_size=int(args.dataset_cache_size),
        load_max_steps=args.time_window_source_max_steps,
    )
    object_ids = dataset.object_ids(args.object_split)
    if not object_ids:
        raise ValueError(f"No objects found in split {args.object_split!r}.")
    if int(args.objects_per_step) > len(object_ids):
        raise ValueError(
            f"--objects-per-step={int(args.objects_per_step)} exceeds the number of "
            f"{args.object_split} objects ({len(object_ids)})."
        )
    log(f"objects in split {args.object_split}: {len(object_ids)}")

    first_collection = dataset.load_object_collection(object_ids[0])
    representative_trajectory = first_collection.trajectories[0]
    args.dt = float(representative_trajectory.timestep)
    args.steps = int(args.query_window_steps)
    positive_batch_capacity = int(args.objects_per_step) * 2 * int(args.query_trajectories_per_view)
    swap_negative_batch_capacity = (
        int(args.objects_per_step) * 2 * int(args.swap_query_trajectories_per_view)
        if float(args.swap_loss_weight) > 0.0
        else 0
    )
    args.batch_capacity = max(positive_batch_capacity + swap_negative_batch_capacity, 1)
    args.max_steps = int(args.query_window_steps)
    args.random_time_windows = bool(args.random_query_windows)
    args.friction_parameterization = f"trajectory-conditioned-{args.decoder_conditioning}"

    log(
        f"building Newton diff scene device={args.device} steps={args.steps} "
        f"batch_capacity={args.batch_capacity} dt={args.dt:.6f}"
    )
    diff_scene = build_diff_scene(args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()

    log("computing active surface contact mask")
    active_indices = compute_active_indices_for_training(
        dataset=dataset,
        object_ids=object_ids,
        diff_scene=diff_scene,
        args=args,
        rng=rng,
    )
    active_side_ids = compute_piecewise_side_ids(diff_scene.local_surface_points_np, active_indices)
    log(
        f"surface_points={len(diff_scene.local_surface_points_np)} "
        f"active_points={len(active_indices)} "
        f"active_left={int(np.sum(active_side_ids == 0))} "
        f"active_right={int(np.sum(active_side_ids == 1))}"
    )

    torch_device = torch.device(args.torch_device if args.torch_device is not None else str(diff_scene.torch_device))
    if torch_device != diff_scene.torch_device:
        raise ValueError(
            "Direct Torch/Warp friction interop requires --torch-device to match --device. "
            f"Got torch_device={torch_device} and newton_device={diff_scene.torch_device}."
        )
    feature_cache = PointFeatureCache(diff_scene=diff_scene, args=args, torch_device=torch_device)
    first_feature = feature_cache.get(dataset.get_object(object_ids[0]))
    first_visual_dim = 0 if first_feature.visual_features is None else int(first_feature.visual_features.shape[1])
    model = build_model(
        args=args,
        point_feature_dim=int(first_feature.features.shape[1]),
        visual_feature_dim=first_visual_dim,
        torch_device=torch_device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        eps=float(args.adam_eps),
    )
    active_indices_torch = torch.as_tensor(active_indices, dtype=torch.long, device=torch_device)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    object_sampler = ObjectBatchSampler(
        object_ids,
        objects_per_step=int(args.objects_per_step),
        strategy=str(args.object_sampling_strategy),
        rng=rng,
    )
    history: list[dict] = []
    best_loss = float("inf")
    best_iteration = 0
    appearance_counts = {str(object_id): 0 for object_id in object_ids}
    start_iteration = 1
    resume_metadata: dict | None = None
    if resume_checkpoint_path is not None:
        resume_metadata = load_resume_checkpoint(
            path=resume_checkpoint_path,
            args=args,
            model=model,
            optimizer=optimizer,
            object_sampler=object_sampler,
            object_ids=object_ids,
            active_indices=active_indices,
            diff_scene=diff_scene,
            rng=rng,
            torch_device=torch_device,
        )
        history = list(resume_metadata["history"])
        best_loss = float(resume_metadata["best_loss"])
        best_iteration = int(resume_metadata["best_iteration"])
        appearance_counts = dict(resume_metadata["appearance_counts"])
        start_iteration = int(resume_metadata["start_iteration"])
        if args.wandb_resume_id is None and resume_metadata.get("wandb_run_id") is not None:
            args.wandb_resume_id = str(resume_metadata["wandb_run_id"])
        log(
            f"resumed checkpoint={resume_checkpoint_path} completed_iteration="
            f"{int(resume_metadata['completed_iteration'])} target_iteration={int(args.opt_iters)} "
            f"appearance_counts_exact={bool(resume_metadata['appearance_counts_exact'])} "
            f"sampler_state_exact={bool(resume_metadata['sampler_state_exact'])} "
            f"rng_state_exact={bool(resume_metadata['rng_state_exact'])}"
        )
    log(
        f"point_feature_dim={int(first_feature.features.shape[1])} "
        f"visual_feature_dim={first_visual_dim} "
        f"latent_dim={int(args.latent_dim)} trainable_params={trainable_params} "
        f"object_sampling={args.object_sampling_strategy}"
    )

    config_payload = {
        "args": _args_dict(args),
        "manifest": str(manifest_path),
        "object_split": str(args.object_split),
        "object_count": len(object_ids),
        "surface_points": int(len(diff_scene.local_surface_points_np)),
        "active_points": int(len(active_indices)),
        "point_feature_metadata": feature_cache.reference_metadata,
        "resume": None
        if resume_metadata is None
        else _jsonable(
            {
                key: value
                for key, value in resume_metadata.items()
                if key not in {"history", "appearance_counts"}
            }
        ),
        "training_interface_note": "Training only; eval is intentionally not run by this script.",
        "friction_route": (
            f"{args.decoder_conditioning}/{args.decoder_activation} trajectory-conditioned decoder predicts mu from "
            f"{args.decoder_point_feature_mode} point features; "
            f"basis_count={int(args.decoder_basis_count)}; "
            f"basis_base={args.decoder_basis_base_mode}; "
            f"basis_norm={args.decoder_basis_normalization}; "
            f"basis_activation={args.decoder_basis_activation}; "
            f"latent_norm={args.decoder_latent_normalization}; "
            f"raw_limit={args.decoder_raw_limit}; "
            f"dino_to_encoder={bool(args.dino_to_encoder)}; "
            "Newton uses the surface-point friction kernel."
        ),
    }
    write_json(output_dir / "config.json", config_payload)

    wandb_run = init_wandb_run(
        args=args,
        output_dir=output_dir,
        config_payload=config_payload,
        trainable_params=trainable_params,
    )
    if wandb_run is not None:
        log(
            f"W&B enabled project={args.wandb_project} "
            f"run={wandb_run.name} mode={args.wandb_mode}"
        )

    if bool(args.dry_run):
        dry_object_ids = object_sampler.sample()
        sample = sample_training_data_for_object_ids(
            dataset=dataset,
            object_ids=dry_object_ids,
            args=args,
            rng=rng,
        )
        dry_entries = tuple(feature_cache.get(item.object_spec) for item in sample)
        dry_visual_features = stack_visual_features(dry_entries, device=torch_device)
        context_a, mask_a = stack_encoder_batches((item.context_a for item in sample), device=torch_device)
        with torch.no_grad():
            output_a = model.encode_context(
                context_a,
                context_valid_mask=mask_a,
                visual_features=dry_visual_features,
            )
            mu = model.decode_friction(
                first_feature.features,
                output_a.latent[:1],
                active_indices=active_indices_torch,
            )
        log(
            "dry_run_ok "
            f"context_shape={tuple(context_a.shape)} latent_shape={tuple(output_a.latent.shape)} "
            f"active_mu_shape={tuple(mu.shape)}"
        )
        write_json(
            output_dir / "dry_run_summary.json",
            {
                **config_payload,
                "context_shape": list(context_a.shape),
                "latent_shape": list(output_a.latent.shape),
                "active_mu_shape": list(mu.shape),
            },
        )
        if wandb_run is not None:
            wandb_run.summary["dry_run"] = True
            wandb_run.summary["context_shape"] = list(context_a.shape)
            wandb_run.summary["latent_shape"] = list(output_a.latent.shape)
            wandb_run.summary["active_mu_shape"] = list(mu.shape)
            wandb_run.finish()
        return

    checkpoint_last = output_dir / f"{output_dir.name}_last.pt"
    checkpoint_best = output_dir / f"{output_dir.name}_best.pt"
    history_jsonl = output_dir / "train_history.jsonl"
    wandb_run_id = None if wandb_run is None else str(wandb_run.id)
    pending_best_payload: dict | None = None
    best_checkpoint_every = max(int(args.best_checkpoint_every), 1)

    for iteration in range(start_iteration, int(args.opt_iters) + 1):
        iteration_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        should_log_iteration = (
            iteration == start_iteration
            or iteration % max(int(args.log_every), 1) == 0
            or iteration == int(args.opt_iters)
        )

        selected_object_ids = object_sampler.sample()
        samples = sample_training_data_for_object_ids(
            dataset=dataset,
            object_ids=selected_object_ids,
            args=args,
            rng=rng,
        )
        for object_id in selected_object_ids:
            appearance_counts[str(object_id)] += 1
        selected_entries = tuple(feature_cache.get(sample.object_spec) for sample in samples)
        visual_features = stack_visual_features(selected_entries, device=torch_device)
        context_a, mask_a = stack_encoder_batches((item.context_a for item in samples), device=torch_device)
        context_b, mask_b = stack_encoder_batches((item.context_b for item in samples), device=torch_device)
        latent_output_a = model.encode_context(
            context_a,
            context_valid_mask=mask_a,
            visual_features=visual_features,
        )
        latent_output_b = model.encode_context(
            context_b,
            context_valid_mask=mask_b,
            visual_features=visual_features,
        )
        regularization = latent_regularization_losses(
            latent_output_a,
            latent_output_b,
            consistency_weight=float(args.consistency_weight),
            contrastive_weight=float(args.contrastive_weight),
            temperature=float(args.contrastive_temperature),
        )

        rollout_stats, view_records, object_records, swap_stats = rollout_all_views_with_reused_swap_and_backpropagate(
            model=model,
            feature_cache=feature_cache,
            samples=samples,
            latent_output_a=latent_output_a,
            latent_output_b=latent_output_b,
            diff_scene=diff_scene,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
            active_indices=active_indices,
            active_indices_torch=active_indices_torch,
            args=args,
            torch_device=torch_device,
            collect_records=should_log_iteration,
        )

        regularization.total.backward()
        if args.grad_clip_norm is not None and float(args.grad_clip_norm) > 0.0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
            )
        else:
            grad_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm_sq += float(torch.sum(param.grad.detach() ** 2).cpu())
            grad_norm = math.sqrt(max(grad_norm_sq, 0.0))
        optimizer.step()

        rollout_loss = float(rollout_stats.get("loss", float("nan")))
        swap_weighted_loss = float(swap_stats.get("weighted_loss", 0.0))
        total_loss = rollout_loss + swap_weighted_loss + float(regularization.total.detach().cpu())
        latent_stats = summarize_latents(latent_output_a.latent, latent_output_b.latent)
        object_sampling_stats = summarize_object_sampling(
            all_object_ids=object_ids,
            selected_object_ids=selected_object_ids,
            appearance_counts=appearance_counts,
            iteration=iteration,
            objects_per_step=int(args.objects_per_step),
        )
        for object_record in object_records:
            object_id = str(object_record.get("object_id", ""))
            object_record["sampling"] = {
                "appearance_count": int(appearance_counts.get(object_id, 0)),
                "appearance_fraction_of_steps": float(appearance_counts.get(object_id, 0)) / float(iteration),
            }
        record = {
            "iteration": int(iteration),
            "total_loss": float(total_loss),
            "rollout_loss": float(rollout_loss),
            "swap_loss": float(swap_stats.get("loss", 0.0)),
            "swap_weighted_loss": float(swap_weighted_loss),
            "regularization_total": float(regularization.total.detach().cpu()),
            "consistency_loss": float(regularization.consistency.detach().cpu()),
            "contrastive_loss": float(regularization.contrastive.detach().cpu()),
            "latent_norm_loss": float(regularization.latent_norm.detach().cpu()),
            "torch_grad_norm_before_clip": float(grad_norm),
            "elapsed_sec": float(time.time() - iteration_start),
            "objects": [sample.object_spec.object_id for sample in samples],
            "object_metrics": object_records,
            "view_metrics": view_records,
            "rollout": rollout_stats,
            "swap": swap_stats,
            "latent": latent_stats,
            "object_sampling": object_sampling_stats,
        }
        if should_log_iteration:
            history.append(record)
            append_jsonl(history_jsonl, record)

        if total_loss < best_loss:
            best_loss = float(total_loss)
            best_iteration = int(iteration)
            best_payload = checkpoint_payload(
                args=args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                active_indices=active_indices,
                diff_scene=diff_scene,
                feature_metadata=feature_cache.reference_metadata,
                history=history,
                best_loss=best_loss,
                best_iteration=best_iteration,
                object_ids=object_ids,
                appearance_counts=appearance_counts,
                object_sampler=object_sampler,
                rng=rng,
                wandb_run_id=wandb_run_id,
            )
            if (
                best_checkpoint_every <= 1
                or iteration % best_checkpoint_every == 0
                or iteration == int(args.opt_iters)
            ):
                torch.save(best_payload, checkpoint_best)
                pending_best_payload = None
            else:
                pending_best_payload = clone_checkpoint_payload_for_deferred_save(best_payload)

        if (
            pending_best_payload is not None
            and (
                iteration % best_checkpoint_every == 0
                or iteration == int(args.opt_iters)
            )
        ):
            torch.save(pending_best_payload, checkpoint_best)
            pending_best_payload = None

        if wandb_run is not None and should_log_iteration:
            wandb_run.log(
                build_wandb_log_payload(
                    record,
                    best_loss=best_loss,
                    best_iteration=best_iteration,
                    detail=str(args.wandb_log_detail),
                ),
                step=iteration,
            )

        if int(args.checkpoint_every) > 0 and (
            iteration % int(args.checkpoint_every) == 0 or iteration == int(args.opt_iters)
        ):
            save_checkpoint(
                path=checkpoint_last,
                args=args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                active_indices=active_indices,
                diff_scene=diff_scene,
                feature_metadata=feature_cache.reference_metadata,
                history=history,
                best_loss=best_loss,
                best_iteration=best_iteration,
                object_ids=object_ids,
                appearance_counts=appearance_counts,
                object_sampler=object_sampler,
                rng=rng,
                wandb_run_id=wandb_run_id,
            )

        if should_log_iteration:
            log(
                f"iter={iteration:04d} total={total_loss:.6f} rollout={rollout_loss:.6f} "
                f"swap={swap_weighted_loss:.6f} "
                f"reg={float(regularization.total.detach().cpu()):.6f} "
                f"cons={float(regularization.consistency.detach().cpu()):.6f} "
                f"nce={float(regularization.contrastive.detach().cpu()):.6f} "
                f"znorm={float(regularization.latent_norm.detach().cpu()):.6f} "
                f"mu_mean={rollout_stats.get('mu_mean', float('nan')):.6f} "
                f"mu_min={rollout_stats.get('mu_min', float('nan')):.6f} "
                f"mu_max={rollout_stats.get('mu_max', float('nan')):.6f} "
                f"swap_gap={swap_stats.get('loss_gap', float('nan')):.6f} "
                f"swap_acc={swap_stats.get('accuracy', float('nan')):.3f} "
                f"appear_min={object_sampling_stats['min_appearance_count']} "
                f"appear_max={object_sampling_stats['max_appearance_count']} "
                f"grad={grad_norm:.6f} elapsed={record['elapsed_sec']:.2f}s"
            )

    if pending_best_payload is not None:
        torch.save(pending_best_payload, checkpoint_best)

    save_checkpoint(
        path=checkpoint_last,
        args=args,
        iteration=int(args.opt_iters),
        model=model,
        optimizer=optimizer,
        active_indices=active_indices,
        diff_scene=diff_scene,
        feature_metadata=feature_cache.reference_metadata,
        history=history,
        best_loss=best_loss,
        best_iteration=best_iteration,
        object_ids=object_ids,
        appearance_counts=appearance_counts,
        object_sampler=object_sampler,
        rng=rng,
        wandb_run_id=wandb_run_id,
    )
    export_preview_point_clouds(
        model=model,
        feature_cache=feature_cache,
        dataset=dataset,
        object_ids=object_ids,
        active_indices=active_indices,
        diff_scene=diff_scene,
        args=args,
        output_dir=output_dir,
        torch_device=torch_device,
    )
    summary = {
        "best_loss": float(best_loss),
        "best_iteration": int(best_iteration),
        "completed_iterations": int(args.opt_iters),
        "last_checkpoint": str(checkpoint_last),
        "best_checkpoint": str(checkpoint_best),
        "history_jsonl": str(history_jsonl),
        "object_appearance_counts": appearance_counts,
        "resumed_from": None if resume_checkpoint_path is None else str(resume_checkpoint_path),
        "start_iteration": int(start_iteration),
        "config": config_payload,
        "final_record": history[-1] if history else None,
    }
    write_json(output_dir / "training_summary.json", summary)
    if wandb_run is not None:
        wandb_run.summary["best_loss"] = float(best_loss)
        wandb_run.summary["best_iteration"] = int(best_iteration)
        wandb_run.summary["completed_iterations"] = int(args.opt_iters)
        wandb_run.summary["last_checkpoint"] = str(checkpoint_last)
        wandb_run.summary["best_checkpoint"] = str(checkpoint_best)
        wandb_run.summary["history_jsonl"] = str(history_jsonl)
        wandb_run.summary["surface_points"] = int(len(diff_scene.local_surface_points_np))
        wandb_run.summary["active_points"] = int(len(active_indices))
        wandb_run.summary["object_sampling_strategy"] = str(args.object_sampling_strategy)
        wandb_run.summary["resumed_from"] = (
            None if resume_checkpoint_path is None else str(resume_checkpoint_path)
        )
        wandb_run.summary["start_iteration"] = int(start_iteration)
        counts = np.asarray(list(appearance_counts.values()), dtype=np.float64)
        if counts.size:
            wandb_run.summary["min_object_appearance_count"] = int(np.min(counts))
            wandb_run.summary["max_object_appearance_count"] = int(np.max(counts))
            wandb_run.summary["mean_object_appearance_count"] = float(np.mean(counts))
        wandb_run.finish()
    log(f"training_complete best_loss={best_loss:.6f} best_iteration={best_iteration}")
    log(f"last_checkpoint={checkpoint_last}")
    log(f"best_checkpoint={checkpoint_best}")


if __name__ == "__main__":
    main()
