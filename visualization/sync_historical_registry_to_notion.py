from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from sync_eval_to_notion import (
    DEFAULT_DATABASE_VERSION,
    DEFAULT_DATA_SOURCE_VERSION,
    MAX_RICH_TEXT,
    NotionClient,
    add_if_present,
    code_block,
    find_property,
    find_title_property,
    paragraph,
    property_type,
    property_value,
    repo_rel,
    skipped_status,
    truncate_text,
    utc_now_iso,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_JSON = ROOT / "eval" / "historical_experiment_registry" / "registry.json"


PROPERTY_ALIASES = {
    "sync_key": ("Sync Key", "Experiment ID", "实验ID", "同步ID"),
    "created_at": ("Created At", "Time", "时间", "训练时间", "开始时间"),
    "status": ("Status", "状态"),
    "family": ("Family", "Experiment Family", "类别", "实验族", "方法族"),
    "experiment": ("Experiment", "Experiment Name", "实验", "实验名称", "名称"),
    "wandb_run_id": ("W&B Run ID", "W&B ID", "WandB Run ID", "wandb_run_id"),
    "wandb_run_name": ("W&B Run Name", "WandB Run Name", "wandb_run_name"),
    "wandb_project": ("W&B Project", "WandB Project", "wandb_project"),
    "wandb_group": ("W&B Group", "WandB Group", "wandb_group"),
    "wandb_path": ("W&B Path", "W&B 路径", "WandB Path", "wandb_path"),
    "wandb_runtime_seconds": ("W&B Runtime Seconds", "Runtime Seconds", "运行秒数"),
    "git_commit": ("Git Commit", "Commit", "git_commit"),
    "artifact": ("Artifact", "Checkpoint", "Checkpoint Path", "检查点", "产物路径"),
    "canonical_checkpoint_role": ("Canonical Checkpoint", "Canonical CKPT", "主检查点"),
    "best_checkpoint_path": ("Best Checkpoint Path", "Best CKPT 路径"),
    "last_checkpoint_path": ("Last Checkpoint Path", "Last CKPT 路径"),
    "output_dir": ("Output Dir", "Experiment Dir", "输出目录", "实验目录"),
    "checkpoint_type": ("Checkpoint Type", "类型", "模型类型"),
    "parameterization": ("Parameterization", "Friction Parameterization", "参数化", "摩擦参数化"),
    "train_dataset": ("Train Dataset", "Training Dataset", "训练集", "训练数据集"),
    "train_dataset_short": ("Train Dataset Short", "Dataset Label", "训练集简称"),
    "max_steps": ("Max Steps", "训练步数", "窗口步数"),
    "random_time_windows": ("Random Time Windows", "随机窗口"),
    "iteration": ("Iteration", "Best Iteration", "迭代"),
    "completed_iteration": ("Completed Iteration", "完成迭代"),
    "target_iteration": ("Target Iteration", "目标迭代"),
    "best_loss": ("Best Loss", "Best Train Loss", "训练Best Loss", "最优训练Loss"),
    "final_loss": ("Final Loss", "Final Train Loss", "最终训练Loss"),
    "best_val_loss": ("Best Val Loss", "验证Best Loss"),
    "rotation68_loss": ("Rotation68 Loss", "rotation68 Eval Loss", "rotation68", "Rotation68"),
    "very_long20_loss": ("Very Long20 Loss", "very_long20 Eval Loss", "very_long20", "VeryLong20"),
    "best_rotation68_loss": ("Best rotation68 Loss",),
    "best_very_long20_loss": ("Best very_long20 Loss",),
    "last_rotation68_loss": ("Last rotation68 Loss",),
    "last_very_long20_loss": ("Last very_long20 Loss",),
    "long20_loss": ("Long20 Loss", "long20", "Long20"),
    "eval_summary_paths": ("Eval Summary Paths", "Eval Summary", "评估摘要", "评估来源"),
    "html_paths": ("HTML Paths", "Visualization", "可视化"),
    "key_config": ("Key Config", "关键配置"),
    "key_metrics": ("Key Metrics", "关键指标"),
    "notes": ("Notes", "备注"),
}


class HistoricalNotionConfig:
    def __init__(self) -> None:
        self.token = os.environ.get("NOTION_TOKEN") or os.environ.get("NOTION_API_KEY")
        self.data_source_id = (
            os.environ.get("NOTION_EXPERIMENT_DATA_SOURCE_ID")
            or os.environ.get("NOTION_HISTORICAL_DATA_SOURCE_ID")
        )
        self.database_id = (
            os.environ.get("NOTION_EXPERIMENT_DATABASE_ID")
            or os.environ.get("NOTION_HISTORICAL_DATABASE_ID")
        )
        if self.data_source_id:
            self.mode = "data_source"
            self.container_id = self.data_source_id
            self.version = os.environ.get("NOTION_VERSION", DEFAULT_DATA_SOURCE_VERSION)
        else:
            self.mode = "database"
            self.container_id = self.database_id
            self.version = os.environ.get("NOTION_VERSION", DEFAULT_DATABASE_VERSION)

    @property
    def missing_reason(self) -> str | None:
        if not self.token:
            return "missing NOTION_TOKEN or NOTION_API_KEY"
        if not self.container_id:
            return (
                "missing NOTION_EXPERIMENT_DATA_SOURCE_ID/NOTION_HISTORICAL_DATA_SOURCE_ID "
                "or NOTION_EXPERIMENT_DATABASE_ID/NOTION_HISTORICAL_DATABASE_ID"
            )
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--registry-json", type=Path, default=DEFAULT_REGISTRY_JSON)
    parser.add_argument(
        "--fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return a non-zero exit code if Notion sync is not configured or a request fails.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only sync the first N records. Useful for testing a new Notion database schema.",
    )
    parser.add_argument(
        "--record-id",
        action="append",
        default=None,
        help="Only sync the exact registry record ID. Can be repeated.",
    )
    return parser.parse_args()


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_registry(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def update_registry_sync_status(path: Path, status: dict[str, Any]) -> None:
    payload = load_registry(path)
    payload["notion_sync"] = status
    write_registry(path, payload)


def compact_list(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item)
    return str(value)


def record_loss(record: dict[str, Any], eval_name: str) -> float | None:
    metrics = record.get("eval_metrics") or {}
    value = (metrics.get(eval_name) or {}).get("loss")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def checkpoint_path_for_role(record: dict[str, Any], role: str) -> str:
    artifact = str(record.get("artifact_path") or "")
    canonical_role = str(record.get("canonical_checkpoint_role") or "primary")
    if canonical_role == role:
        return artifact
    alternatives = [str(path) for path in record.get("alternate_artifact_paths") or []]
    if role == "last":
        return next((path for path in alternatives if Path(path).stem.lower().endswith("_last")), "")
    return next((path for path in alternatives if not Path(path).stem.lower().endswith("_last")), "")


def protocol_rank(eval_name: str) -> tuple[int, str]:
    lower = eval_name.lower()
    if "diagnostic" in lower or "probe" in lower or "legacy" in lower:
        return (50, eval_name)
    if "progress" in lower:
        return (40, eval_name)
    if "gain" in lower:
        return (10, eval_name)
    return (20, eval_name)


def checkpoint_loss_for_role(record: dict[str, Any], role: str, dataset: str) -> float | None:
    if str(record.get("canonical_checkpoint_role") or "primary") == role:
        return record_loss(record, dataset)
    path = checkpoint_path_for_role(record, role)
    metrics = (record.get("alternate_eval_metrics") or {}).get(path) or {}
    candidates = [
        (protocol_rank(str(eval_name)), metric)
        for eval_name, metric in metrics.items()
        if isinstance(metric, dict) and str(metric.get("dataset_label") or "") == dataset
    ]
    if not candidates:
        return None
    value = sorted(candidates, key=lambda item: item[0])[0][1].get("loss")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def record_sync_key(record: dict[str, Any]) -> str:
    return str(record.get("record_id") or record.get("output_dir") or record.get("artifact_path") or "")


def record_title(record: dict[str, Any]) -> str:
    return str(record.get("experiment_name") or record_sync_key(record))


def train_dataset_short(record: dict[str, Any]) -> str:
    value = str(record.get("train_dataset") or "")
    if "very_long_rotation_friction_diagnostics_l0p20_r0p50_2000" in value:
        return "very_long2000"
    if "rotation_friction_diagnostics_l0p20_r0p50_2000" in value:
        return "rotation2000"
    if "very_long_rotation_friction_diagnostics_l0p20_r0p50_20" in value:
        return "very_long20"
    if "rotation_friction_diagnostics_l0p20_r0p50_68" in value:
        return "rotation68"
    return Path(value).stem if value else ""


def notion_family(record: dict[str, Any]) -> str:
    family = str(record.get("family") or "")
    return "RNN_residual" if family == "rnn_residual" else family


def key_config_text(record: dict[str, Any]) -> str:
    fields = (
        "canonical_checkpoint_role",
        "checkpoint_type",
        "parameterization",
        "max_steps",
        "random_time_windows",
        "iteration",
        "completed_iteration",
        "target_iteration",
        "history_window_steps",
        "prediction_window_steps",
        "residual_output_mode",
        "direct_state_output_mode",
        "model_semantics",
        "condition_formula",
        "training_phase",
        "training_residual_gain",
    )
    values = {field: record.get(field) for field in fields if record.get(field) not in (None, "", [])}
    selected = record.get("selected_eval_protocols")
    if selected:
        values["selected_eval_protocols"] = selected
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def key_metrics_text(record: dict[str, Any]) -> str:
    parts = [
        f"canonical checkpoint={record.get('canonical_checkpoint_role') or 'primary'}: {record.get('artifact_path')}"
    ]
    if record.get("best_loss") is not None:
        parts.append(f"best train loss={record.get('best_loss')}")
    if record.get("final_loss") is not None:
        parts.append(f"final train loss={record.get('final_loss')}")
    selected = record.get("selected_eval_protocols") or {}
    for dataset in ("rotation68", "very_long20"):
        loss = record_loss(record, dataset)
        if loss is not None:
            parts.append(f"{dataset}={loss} using {selected.get(dataset, dataset)}")
    if record.get("alternate_eval_metrics"):
        parts.append("alternate checkpoint evals are retained for sensitivity analysis and are not mixed into main metrics")
    if notion_family(record) in {"RNN_residual", "pointnet_residual", "residual_mlp", "stateful_gru_residual"}:
        parts.append("residual model is simulator-mismatch correction, not direct physical friction identification")
    if notion_family(record) == "stateful_gru_direct_state":
        parts.append(
            "direct-state model predicts a MuJoCo-like state trajectory conditioned on Newton open-loop states; "
            "it is not direct physical friction identification"
        )
    return "; ".join(parts)


def add_record_property(
    *,
    properties: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    field: str,
    value: object,
    allowed: set[str] | None = None,
) -> None:
    aliases = PROPERTY_ALIASES[field]
    prop_name = find_property(schema, aliases, allowed=allowed)
    if not prop_name:
        return
    prop_value = property_value(schema[prop_name], value)
    if prop_value is not None:
        properties[prop_name] = prop_value


def build_properties(record: dict[str, Any], schema: dict[str, dict[str, Any]]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    title_property = find_title_property(schema)
    properties[title_property] = property_value(schema[title_property], record_title(record))
    add_record_property(properties=properties, schema=schema, field="sync_key", value=record_sync_key(record), allowed={"rich_text", "title"})
    add_record_property(properties=properties, schema=schema, field="created_at", value=record.get("created_at"), allowed={"date", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="status", value=record.get("status"), allowed={"select", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="family", value=notion_family(record), allowed={"select", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="experiment", value=record.get("experiment_name"), allowed={"rich_text", "title"})
    add_record_property(properties=properties, schema=schema, field="wandb_run_id", value=record.get("wandb_run_id"), allowed={"rich_text"})
    add_record_property(properties=properties, schema=schema, field="wandb_run_name", value=record.get("wandb_run_name"), allowed={"rich_text", "title"})
    add_record_property(properties=properties, schema=schema, field="wandb_project", value=record.get("wandb_project"), allowed={"select", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="wandb_group", value=record.get("wandb_group"), allowed={"select", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="wandb_path", value=record.get("wandb_path"), allowed={"rich_text", "url"})
    add_record_property(properties=properties, schema=schema, field="wandb_runtime_seconds", value=record.get("wandb_runtime_seconds"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="git_commit", value=record.get("git_commit"), allowed={"rich_text"})
    add_record_property(properties=properties, schema=schema, field="artifact", value=record.get("artifact_path"), allowed={"rich_text", "url"})
    add_record_property(properties=properties, schema=schema, field="canonical_checkpoint_role", value=record.get("canonical_checkpoint_role"), allowed={"select", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="best_checkpoint_path", value=checkpoint_path_for_role(record, "best"), allowed={"rich_text", "url"})
    add_record_property(properties=properties, schema=schema, field="last_checkpoint_path", value=checkpoint_path_for_role(record, "last"), allowed={"rich_text", "url"})
    add_record_property(properties=properties, schema=schema, field="output_dir", value=record.get("output_dir"), allowed={"rich_text", "url"})
    add_record_property(properties=properties, schema=schema, field="checkpoint_type", value=record.get("checkpoint_type"), allowed={"select", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="parameterization", value=record.get("parameterization"), allowed={"select", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="train_dataset", value=record.get("train_dataset"), allowed={"rich_text", "url"})
    add_record_property(properties=properties, schema=schema, field="train_dataset_short", value=train_dataset_short(record), allowed={"select", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="max_steps", value=record.get("max_steps"), allowed={"number", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="random_time_windows", value=record.get("random_time_windows"), allowed={"checkbox", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="iteration", value=record.get("iteration"), allowed={"number", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="completed_iteration", value=record.get("completed_iteration"), allowed={"number", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="target_iteration", value=record.get("target_iteration"), allowed={"number", "rich_text"})
    add_record_property(properties=properties, schema=schema, field="best_loss", value=record.get("best_loss"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="final_loss", value=record.get("final_loss"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="best_val_loss", value=record.get("best_val_loss"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="rotation68_loss", value=record_loss(record, "rotation68"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="very_long20_loss", value=record_loss(record, "very_long20"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="best_rotation68_loss", value=checkpoint_loss_for_role(record, "best", "rotation68"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="best_very_long20_loss", value=checkpoint_loss_for_role(record, "best", "very_long20"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="last_rotation68_loss", value=checkpoint_loss_for_role(record, "last", "rotation68"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="last_very_long20_loss", value=checkpoint_loss_for_role(record, "last", "very_long20"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="long20_loss", value=record_loss(record, "long20"), allowed={"number"})
    add_record_property(properties=properties, schema=schema, field="eval_summary_paths", value=compact_list(record.get("eval_summary_paths")), allowed={"rich_text", "url"})
    add_record_property(properties=properties, schema=schema, field="html_paths", value=compact_list(record.get("html_paths")), allowed={"rich_text", "url"})
    add_record_property(properties=properties, schema=schema, field="key_config", value=key_config_text(record), allowed={"rich_text"})
    add_record_property(properties=properties, schema=schema, field="key_metrics", value=key_metrics_text(record), allowed={"rich_text"})
    add_record_property(properties=properties, schema=schema, field="notes", value=record.get("notes"), allowed={"rich_text"})
    return properties


def filter_for_existing_page(record: dict[str, Any], schema: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sync_prop = find_property(schema, PROPERTY_ALIASES["sync_key"], allowed={"rich_text", "title"})
    if sync_prop:
        kind = property_type(schema[sync_prop])
        if kind == "title":
            return {"property": sync_prop, "title": {"equals": record_sync_key(record)}}
        return {"property": sync_prop, "rich_text": {"equals": record_sync_key(record)}}
    title_prop = find_title_property(schema)
    return {"property": title_prop, "title": {"equals": record_title(record)}}


def build_children(record: dict[str, Any]) -> list[dict[str, Any]]:
    compact = {
        "record_id": record.get("record_id"),
        "status": record.get("status"),
        "family": record.get("family"),
        "experiment_name": record.get("experiment_name"),
        "wandb_run_id": record.get("wandb_run_id"),
        "wandb_run_name": record.get("wandb_run_name"),
        "wandb_project": record.get("wandb_project"),
        "wandb_group": record.get("wandb_group"),
        "wandb_path": record.get("wandb_path"),
        "wandb_runtime_seconds": record.get("wandb_runtime_seconds"),
        "git_commit": record.get("git_commit"),
        "artifact_path": record.get("artifact_path"),
        "output_dir": record.get("output_dir"),
        "checkpoint_type": record.get("checkpoint_type"),
        "parameterization": record.get("parameterization"),
        "train_dataset": record.get("train_dataset"),
        "max_steps": record.get("max_steps"),
        "random_time_windows": record.get("random_time_windows"),
        "iteration": record.get("iteration"),
        "best_loss": record.get("best_loss"),
        "final_loss": record.get("final_loss"),
        "eval_metrics": record.get("eval_metrics"),
        "eval_summary_paths": record.get("eval_summary_paths"),
        "html_paths": record.get("html_paths"),
        "alternate_artifact_paths": record.get("alternate_artifact_paths"),
    }
    json_text = json.dumps(compact, indent=2, ensure_ascii=False, sort_keys=True)
    chunks = [json_text[idx : idx + MAX_RICH_TEXT] for idx in range(0, len(json_text), MAX_RICH_TEXT)]
    children = [
        paragraph(f"Experiment: {record_title(record)}"),
        paragraph(f"Family: {record.get('family')} | Status: {record.get('status')}"),
        paragraph(
            "Loss: "
            f"train_best={record.get('best_loss')}, "
            f"rotation68={record_loss(record, 'rotation68')}, "
            f"very_long20={record_loss(record, 'very_long20')}"
        ),
    ]
    children.extend(code_block(chunk) for chunk in chunks[:4])
    return children


def sync_one_record(client: NotionClient, schema: dict[str, dict[str, Any]], record: dict[str, Any]) -> dict[str, Any]:
    properties = build_properties(record, schema)
    query = client.query_container(filter_for_existing_page(record, schema))
    results = query.get("results", [])
    if results:
        page = client.update_page(str(results[0]["id"]), properties)
        action = "updated"
    else:
        page = client.create_page(properties, build_children(record))
        action = "created"
    return {
        "record_id": record_sync_key(record),
        "status": action,
        "page_id": page.get("id"),
        "page_url": page.get("url"),
    }


def sync_registry_to_notion(
    registry_path: Path,
    *,
    fail_on_error: bool = False,
    limit: int | None = None,
    record_ids: list[str] | None = None,
) -> dict[str, Any]:
    payload = load_registry(registry_path)
    records = payload.get("records") or []
    if record_ids:
        selected = set(record_ids)
        records = [record for record in records if record_sync_key(record) in selected]
    if limit is not None:
        records = records[:limit]
    if not records:
        return {"status": "skipped", "reason": "no registry records", "results": []}

    config = HistoricalNotionConfig()
    missing_reason = config.missing_reason
    if missing_reason is not None:
        status = skipped_status(missing_reason)
        update_registry_sync_status(registry_path, status)
        if fail_on_error:
            raise RuntimeError(missing_reason)
        return {"status": "skipped", "reason": missing_reason, "results": []}

    client = NotionClient(config)
    try:
        container = client.retrieve_container()
        schema = container.get("properties", {})
        if not schema:
            raise ValueError("Notion database/data source returned no properties schema")
        results = [sync_one_record(client, schema, record) for record in records]
        status = {
            "status": "ok",
            "synced_at_utc": utc_now_iso(),
            "record_count": len(results),
            "container_mode": config.mode,
            "results": results,
        }
        update_registry_sync_status(registry_path, status)
        return status
    except Exception as exc:
        status = {"status": "error", "reason": str(exc), "synced_at_utc": utc_now_iso(), "results": []}
        update_registry_sync_status(registry_path, status)
        if fail_on_error:
            raise
        return status


def main() -> None:
    args = parse_args()
    result = sync_registry_to_notion(
        args.registry_json,
        fail_on_error=bool(args.fail_on_error),
        limit=args.limit,
        record_ids=args.record_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
