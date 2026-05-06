from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from project_paths import DEFAULT_DEMO_PATH, DEFAULT_PLY_PATH
from pbd_math import normalize_quaternion, quaternion_conjugate, quaternion_multiply
from pbd_scene import (
    advance_prescribed_cluster,
    build_scene_from_segmented_ply,
    find_cluster,
    step_scene,
)
from pbd_types import (
    DEFAULT_CONTACT_STIFFNESS,
    DEFAULT_CONTACT_DAMPING,
    DEFAULT_CONTACT_MARGIN,
    DEFAULT_EE_MASS,
    DEFAULT_EE_RADIUS_SCALE,
    DEFAULT_EE_SEG_ID,
    DEFAULT_EE_VOXEL,
    DEFAULT_FRICTION_REGULARIZATION,
    DEFAULT_MAX_VELOCITY,
    DEFAULT_OBJECT_FRICTION,
    DEFAULT_SUBSTEPS,
    DEFAULT_TABLE_FRICTION,
    DEFAULT_TABLE_SEG_ID,
    DEFAULT_TABLE_VOXEL,
    DEFAULT_TEE_MASS,
    DEFAULT_TEE_RADIUS_SCALE,
    DEFAULT_TEE_SEG_ID,
    DEFAULT_TEE_VOXEL,
    DEFAULT_VELOCITY_DAMPING,
)


@dataclass
class DemoSegment:
    actions: np.ndarray
    tee_pose: np.ndarray
    start_step: int
    num_steps: int
    traj_key: str
    metadata: dict


FIT_PARAMETER_SPECS = (
    ("table_friction", "init_table_friction", 0.0),
    ("object_friction", "init_object_friction", 0.0),
    ("contact_margin", "init_contact_margin", 1e-6),
    ("contact_damping", "init_contact_damping", 0.0),
    ("tee_mass", "init_tee_mass", 1e-6),
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def inverse_softplus(value: float, min_value: float = 0.0) -> float:
    shifted = max(value - min_value, 1e-8)
    return math.log(math.expm1(shifted))


def positive_parameter(raw: torch.Tensor, min_value: float = 0.0) -> torch.Tensor:
    return F.softplus(raw) + min_value


def infer_demo_start_step(ply_path: Path) -> int | None:
    match = re.search(r"step_(\d+)", ply_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def load_demo_segment(
    demo_path: Path,
    traj_key: str,
    start_step: int,
    num_steps: int | None,
) -> DemoSegment:
    with h5py.File(demo_path, "r") as f:
        if traj_key not in f:
            raise KeyError(f"Trajectory key '{traj_key}' was not found in {demo_path}")

        traj = f[traj_key]
        raw_actions = np.asarray(traj["actions"][:], dtype=np.float32)
        raw_tee_state = np.asarray(
            traj["env_states"]["actors"]["Tee"][:, :7],
            dtype=np.float32,
        )

    if start_step < 0:
        raise ValueError(f"start_step must be >= 0, got {start_step}")
    if start_step >= len(raw_actions):
        raise ValueError(
            f"start_step={start_step} is outside the action sequence of length {len(raw_actions)}"
        )

    max_steps = len(raw_actions) - start_step
    if num_steps is None:
        segment_steps = max_steps
    else:
        segment_steps = min(max(int(num_steps), 1), max_steps)

    actions = raw_actions[start_step : start_step + segment_steps].copy()
    tee_pose = raw_tee_state[start_step : start_step + segment_steps + 1].copy()
    if len(tee_pose) != segment_steps + 1:
        raise RuntimeError(
            f"Expected {segment_steps + 1} Tee poses, got {len(tee_pose)}. "
            f"Check the demo file and start_step={start_step}."
        )

    metadata_path = demo_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return DemoSegment(
        actions=actions,
        tee_pose=tee_pose,
        start_step=start_step,
        num_steps=segment_steps,
        traj_key=traj_key,
        metadata=metadata,
    )


def align_sim_pose_trajectory(
    sim_pose_xyzw: torch.Tensor,
    demo_pose_xyzw: torch.Tensor,
) -> torch.Tensor:
    sim_pose = sim_pose_xyzw.clone()
    demo_pose = demo_pose_xyzw.clone()

    sim_pos_0 = sim_pose[0, :3]
    demo_pos_0 = demo_pose[0, :3]
    position_offset = demo_pos_0 - sim_pos_0

    sim_quat_0 = normalize_quaternion(sim_pose[0, 3:7])
    demo_quat_0 = normalize_quaternion(demo_pose[0, 3:7])
    quat_offset = normalize_quaternion(
        quaternion_multiply(demo_quat_0.unsqueeze(0), quaternion_conjugate(sim_quat_0).unsqueeze(0))
    ).squeeze(0)

    aligned = sim_pose.clone()
    aligned[:, :3] = aligned[:, :3] + position_offset.unsqueeze(0)
    aligned[:, 3:7] = normalize_quaternion(
        quaternion_multiply(
            quat_offset.unsqueeze(0).expand(len(sim_pose), -1),
            normalize_quaternion(sim_pose[:, 3:7]),
        )
    )
    return aligned


def quaternion_mse_with_sign_alignment(
    pred_quat_xyzw: torch.Tensor,
    target_quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    pred_quat = normalize_quaternion(pred_quat_xyzw)
    target_quat = normalize_quaternion(target_quat_xyzw)
    dot = torch.sum(pred_quat * target_quat, dim=-1, keepdim=True)
    sign = torch.where(dot < 0.0, -1.0, 1.0)
    pred_aligned = pred_quat * sign
    return torch.mean((pred_aligned - target_quat) ** 2)


def tee_pose_loss(
    sim_pose_xyzw: torch.Tensor,
    demo_pose_xyzw: torch.Tensor,
    position_weight: float,
    orientation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aligned_sim = align_sim_pose_trajectory(sim_pose_xyzw, demo_pose_xyzw)
    pos_loss = torch.mean((aligned_sim[:, :3] - demo_pose_xyzw[:, :3]) ** 2)
    ori_loss = quaternion_mse_with_sign_alignment(
        aligned_sim[:, 3:7],
        demo_pose_xyzw[:, 3:7],
    )
    total_loss = position_weight * pos_loss + orientation_weight * ori_loss
    return total_loss, pos_loss, ori_loss


def init_wandb(args: argparse.Namespace, demo: DemoSegment) -> Any | None:
    if not args.wandb:
        return None

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb logging was requested, but the 'wandb' package is not installed. "
            "Install it with `pip install wandb`."
        ) from exc

    config = {key: to_jsonable(value) for key, value in vars(args).items()}
    config["demo_loaded"] = dict(
        demo_path=str(args.demo_path),
        traj_key=demo.traj_key,
        start_step=demo.start_step,
        num_steps=demo.num_steps,
        metadata=to_jsonable(demo.metadata),
    )

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        dir=str(args.wandb_dir) if args.wandb_dir is not None else None,
        tags=args.wandb_tags,
        group=args.wandb_group,
        mode=args.wandb_mode,
        config=config,
        save_code=True,
    )


def resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_raw_fit_parameters(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.nn.Parameter]:
    raw_params: dict[str, torch.nn.Parameter] = {}
    for param_name, init_arg_name, min_value in FIT_PARAMETER_SPECS:
        raw_params[param_name] = torch.nn.Parameter(
            torch.tensor(
                inverse_softplus(float(getattr(args, init_arg_name)), min_value=min_value),
                dtype=torch.float32,
                device=device,
            )
        )
    return raw_params


def materialize_fit_parameters(
    raw_params: dict[str, torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    for param_name, _, min_value in FIT_PARAMETER_SPECS:
        value = positive_parameter(raw_params[param_name], min_value=min_value)
        value.retain_grad()
        params[param_name] = value
    return params


def get_initial_fit_parameters(args: argparse.Namespace) -> dict[str, float]:
    return {
        param_name: float(getattr(args, init_arg_name))
        for param_name, init_arg_name, _ in FIT_PARAMETER_SPECS
    }


def to_float_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def collect_fit_metrics(
    iteration: int,
    loss: torch.Tensor,
    pos_loss: torch.Tensor,
    ori_loss: torch.Tensor,
    params: dict[str, torch.Tensor],
    initial_params: dict[str, float],
) -> dict[str, float]:
    result = dict(
        iteration=iteration,
        loss=to_float_scalar(loss),
        position_loss=to_float_scalar(pos_loss),
        orientation_loss=to_float_scalar(ori_loss),
    )
    for param_name, _, _ in FIT_PARAMETER_SPECS:
        current_value = to_float_scalar(params[param_name])
        grad = params[param_name].grad
        result[param_name] = current_value
        result[f"{param_name}_delta"] = current_value - initial_params[param_name]
        result[f"{param_name}_grad"] = 0.0 if grad is None else to_float_scalar(grad)
    return result


def build_wandb_log_payload(
    result: dict[str, float],
    best_loss: float,
) -> dict[str, float]:
    return {
        "train/loss": result["loss"],
        "train/position_loss": result["position_loss"],
        "train/orientation_loss": result["orientation_loss"],
        "params/table_friction": result["table_friction"],
        "params/object_friction": result["object_friction"],
        "params/contact_margin": result["contact_margin"],
        "params/contact_damping": result["contact_damping"],
        "params/tee_mass": result["tee_mass"],
        "delta/table_friction": result["table_friction_delta"],
        "delta/object_friction": result["object_friction_delta"],
        "delta/contact_margin": result["contact_margin_delta"],
        "delta/contact_damping": result["contact_damping_delta"],
        "delta/tee_mass": result["tee_mass_delta"],
        "grads/table_friction": result["table_friction_grad"],
        "grads/object_friction": result["object_friction_grad"],
        "grads/contact_margin": result["contact_margin_grad"],
        "grads/contact_damping": result["contact_damping_grad"],
        "grads/tee_mass": result["tee_mass_grad"],
        "train/best_loss": best_loss,
    }


def rollout_tee_pose_trajectory(
    *,
    ply_path: Path,
    demo_actions: torch.Tensor,
    table_seg_id: int,
    tee_seg_id: int,
    ee_seg_id: int,
    table_voxel: float,
    tee_voxel: float,
    ee_voxel: float,
    tee_radius_scale: float,
    ee_radius_scale: float,
    tee_mass: torch.Tensor,
    ee_mass: float,
    xpbd_iterations: int,
    table_friction: torch.Tensor,
    object_friction: torch.Tensor,
    contact_damping: torch.Tensor,
    contact_margin: torch.Tensor,
    contact_stiffness: float,
    friction_regularization: float,
    sim_dt: float,
    substeps: int,
    velocity_damping: float,
    max_velocity: float,
    device: str,
) -> torch.Tensor:
    scene = build_scene_from_segmented_ply(
        ply_path=ply_path,
        table_seg_id=table_seg_id,
        tee_seg_id=tee_seg_id,
        ee_seg_id=ee_seg_id,
        table_voxel=table_voxel,
        tee_voxel=tee_voxel,
        ee_voxel=ee_voxel,
        tee_radius_scale=tee_radius_scale,
        ee_radius_scale=ee_radius_scale,
        tee_mass=tee_mass,
        ee_mass=ee_mass,
        xpbd_iterations=xpbd_iterations,
        table_friction=table_friction,
        object_friction=object_friction,
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
        contact_margin=contact_margin,
        friction_regularization=friction_regularization,
        device=device,
    )

    tee_cluster = find_cluster(scene, "tee")
    ee_cluster = find_cluster(scene, "end_effector")
    tee_pose_steps = [scene.state_0.body_q[tee_cluster.body_id, :7]]

    substep_count = max(int(substeps), 1)
    sub_dt = sim_dt / substep_count
    for ee_action in demo_actions:
        sub_action = ee_action / substep_count
        for _ in range(substep_count):
            advance_prescribed_cluster(
                scene=scene,
                cluster=ee_cluster,
                delta_xyz=sub_action,
                dt=sub_dt,
            )
            step_scene(
                scene=scene,
                dt=sub_dt,
                velocity_damping=velocity_damping,
                max_velocity=max_velocity,
            )
        tee_pose_steps.append(scene.state_0.body_q[tee_cluster.body_id, :7])

    return torch.stack(tee_pose_steps, dim=0)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Fit a small set of PBD physical parameters to a ManiSkill demo by replaying the "
            "same EE action sequence and minimizing Tee pose error."
        )
    )
    parser.add_argument(
        "--demo-path",
        type=Path,
        default=DEFAULT_DEMO_PATH,
    )
    parser.add_argument(
        "--traj-key",
        type=str,
        default="traj_0",
    )
    parser.add_argument(
        "--ply-path",
        type=Path,
        default=DEFAULT_PLY_PATH,
    )
    parser.add_argument(
        "--demo-start-step",
        type=int,
        default=None,
        help=(
            "Demo step to start from. If omitted, infer from the PLY filename like step_0560."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Optional number of demo actions to fit. Defaults to all remaining steps.",
    )
    parser.add_argument("--fit-iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb-project", type=str, default="newton-fit-demo-params")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="tee-system-id")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
    parser.add_argument("--sim-dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--xpbd-iterations", type=int, default=5)
    parser.add_argument("--velocity-damping", type=float, default=DEFAULT_VELOCITY_DAMPING)
    parser.add_argument("--max-velocity", type=float, default=DEFAULT_MAX_VELOCITY)
    parser.add_argument("--contact-stiffness", type=float, default=DEFAULT_CONTACT_STIFFNESS)
    parser.add_argument(
        "--friction-regularization",
        type=float,
        default=DEFAULT_FRICTION_REGULARIZATION,
    )
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=1.0)

    parser.add_argument("--table-seg-id", type=int, default=DEFAULT_TABLE_SEG_ID)
    parser.add_argument("--tee-seg-id", type=int, default=DEFAULT_TEE_SEG_ID)
    parser.add_argument("--ee-seg-id", type=int, default=DEFAULT_EE_SEG_ID)
    parser.add_argument("--table-voxel", type=float, default=DEFAULT_TABLE_VOXEL)
    parser.add_argument("--tee-voxel", type=float, default=DEFAULT_TEE_VOXEL)
    parser.add_argument("--ee-voxel", type=float, default=DEFAULT_EE_VOXEL)
    parser.add_argument("--tee-radius-scale", type=float, default=DEFAULT_TEE_RADIUS_SCALE)
    parser.add_argument("--ee-radius-scale", type=float, default=DEFAULT_EE_RADIUS_SCALE)
    parser.add_argument("--ee-mass", type=float, default=DEFAULT_EE_MASS)

    parser.add_argument("--init-table-friction", type=float, default=DEFAULT_TABLE_FRICTION)
    parser.add_argument("--init-object-friction", type=float, default=DEFAULT_OBJECT_FRICTION)
    parser.add_argument("--init-contact-margin", type=float, default=DEFAULT_CONTACT_MARGIN)
    parser.add_argument("--init-contact-damping", type=float, default=DEFAULT_CONTACT_DAMPING)
    parser.add_argument("--init-tee-mass", type=float, default=DEFAULT_TEE_MASS)

    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional JSON file to save the final fit result.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if not args.demo_path.exists():
        raise FileNotFoundError(
            f"Demo HDF5 not found: {args.demo_path}. "
            "Pass --demo-path or place 20260406_183206.h5 at the repository root."
        )
    if not args.ply_path.exists():
        raise FileNotFoundError(
            f"Segmented PLY not found: {args.ply_path}. "
            "Pass --ply-path or copy pointcloud_step_0560.ply into PushT183206/."
        )

    inferred_start = infer_demo_start_step(args.ply_path)
    demo_start_step = args.demo_start_step
    if demo_start_step is None:
        demo_start_step = inferred_start if inferred_start is not None else 0
    if args.demo_start_step is None and inferred_start is not None:
        print(f"Inferred demo_start_step={demo_start_step} from {args.ply_path.name}")

    device = resolve_device(args.device)
    args.device = str(device)
    print(f"Using device={args.device}")

    demo = load_demo_segment(
        demo_path=args.demo_path,
        traj_key=args.traj_key,
        start_step=demo_start_step,
        num_steps=args.num_steps,
    )

    print(
        f"Loaded {demo.traj_key} from {args.demo_path} | "
        f"start_step={demo.start_step} | num_steps={demo.num_steps}"
    )

    wandb_run = init_wandb(args, demo)
    if wandb_run is not None:
        print(
            f"W&B enabled | project={args.wandb_project} | "
            f"run={wandb_run.name} | mode={args.wandb_mode}"
        )

    raw_params = build_raw_fit_parameters(args, device)
    initial_params = get_initial_fit_parameters(args)
    print(
        "Initial params | "
        f"table_mu={initial_params['table_friction']:.6f} "
        f"object_mu={initial_params['object_friction']:.6f} "
        f"margin={initial_params['contact_margin']:.6f} "
        f"damping={initial_params['contact_damping']:.6f} "
        f"tee_mass={initial_params['tee_mass']:.6f}"
    )

    optimizer = torch.optim.Adam(
        list(raw_params.values()),
        lr=args.lr,
    )

    demo_actions = torch.as_tensor(demo.actions, dtype=torch.float32, device=device)
    demo_tee_pose = torch.as_tensor(demo.tee_pose, dtype=torch.float32, device=device)

    best_loss = None
    best_result = None
    try:
        for iteration in range(1, args.fit_iters + 1):
            optimizer.zero_grad()

            params = materialize_fit_parameters(raw_params)

            sim_tee_pose = rollout_tee_pose_trajectory(
                ply_path=args.ply_path,
                demo_actions=demo_actions,
                table_seg_id=args.table_seg_id,
                tee_seg_id=args.tee_seg_id,
                ee_seg_id=args.ee_seg_id,
                table_voxel=args.table_voxel,
                tee_voxel=args.tee_voxel,
                ee_voxel=args.ee_voxel,
                tee_radius_scale=args.tee_radius_scale,
                ee_radius_scale=args.ee_radius_scale,
                tee_mass=params["tee_mass"],
                ee_mass=args.ee_mass,
                xpbd_iterations=args.xpbd_iterations,
                table_friction=params["table_friction"],
                object_friction=params["object_friction"],
                contact_damping=params["contact_damping"],
                contact_margin=params["contact_margin"],
                contact_stiffness=args.contact_stiffness,
                friction_regularization=args.friction_regularization,
                sim_dt=args.sim_dt,
                substeps=args.substeps,
                velocity_damping=args.velocity_damping,
                max_velocity=args.max_velocity,
                device=args.device,
            )

            loss, pos_loss, ori_loss = tee_pose_loss(
                sim_pose_xyzw=sim_tee_pose,
                demo_pose_xyzw=demo_tee_pose,
                position_weight=args.position_weight,
                orientation_weight=args.orientation_weight,
            )
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(list(raw_params.values()), max_norm=args.grad_clip)

            optimizer.step()

            result = collect_fit_metrics(
                iteration=iteration,
                loss=loss,
                pos_loss=pos_loss,
                ori_loss=ori_loss,
                params=params,
                initial_params=initial_params,
            )
            if best_loss is None or result["loss"] < best_loss:
                best_loss = result["loss"]
                best_result = result

            if wandb_run is not None:
                wandb_run.log(build_wandb_log_payload(result, best_loss), step=iteration)

            if iteration == 1 or iteration % args.log_every == 0 or iteration == args.fit_iters:
                print(
                    f"[{iteration:04d}] "
                    f"loss={result['loss']:.6f} "
                    f"pos={result['position_loss']:.6f} "
                    f"ori={result['orientation_loss']:.6f} "
                    f"table_mu={result['table_friction']:.6f} "
                    f"object_mu={result['object_friction']:.6f} "
                    f"margin={result['contact_margin']:.6f} "
                    f"damping={result['contact_damping']:.6f} "
                    f"tee_mass={result['tee_mass']:.6f}"
                )

        print("Best result:", json.dumps(best_result, indent=2))
        if args.save_json is not None:
            args.save_json.parent.mkdir(parents=True, exist_ok=True)
            args.save_json.write_text(json.dumps(best_result, indent=2))
            print(f"Saved fit result to {args.save_json}")

        if wandb_run is not None and best_result is not None:
            for key, value in best_result.items():
                wandb_run.summary[f"best_{key}"] = value
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
