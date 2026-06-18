from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
NOTION_API_BASE = "https://api.notion.com/v1"
DEFAULT_DATA_SOURCE_VERSION = "2026-03-11"
DEFAULT_DATABASE_VERSION = "2022-06-28"
MAX_RICH_TEXT = 1900


PROPERTY_ALIASES = {
    "sync_key": ("Sync Key", "Eval ID", "评估ID", "同步ID"),
    "time": ("Eval Time", "Time", "时间", "评估时间", "Created At"),
    "status": ("Status", "状态"),
    "experiment": ("Experiment", "Experiment Name", "实验", "实验名称"),
    "method": ("Method", "方法"),
    "eval_name": ("Eval Name", "Eval", "评估名称", "评估集"),
    "dataset": ("Dataset", "数据集"),
    "checkpoint": ("Checkpoint", "Checkpoint Path", "检查点"),
    "checkpoint_type": ("Checkpoint Type", "类型", "模型类型"),
    "parameterization": ("Parameterization", "Friction Parameterization", "参数化", "摩擦参数化"),
    "overlay_loss_mean": ("Overlay Loss Mean", "Eval Loss", "Mean Loss", "平均评估Loss"),
    "overlay_loss_std": ("Overlay Loss Std", "Std Loss", "评估Loss标准差"),
    "overlay_loss_min": ("Overlay Loss Min", "Min Loss", "最小Loss"),
    "overlay_loss_max": ("Overlay Loss Max", "Max Loss", "最大Loss"),
    "trajectory_count": ("Trajectory Count", "Trajectories", "轨迹数"),
    "eval_dir": ("Eval Dir", "评估目录"),
    "summary_path": ("Summary Path", "Summary JSON", "摘要路径"),
    "rollout_path": ("Rollout Path", "Trajectory Rollouts", "轨迹文件"),
    "html_output_dir": ("HTML Output Dir", "HTML目录"),
    "residual_gain": ("Residual Gain", "PointNet Residual Gain", "增益"),
    "residual_output_mode": ("Residual Output Mode", "Output Mode", "输出模式"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--summary-json",
        type=Path,
        action="append",
        default=None,
        help="Eval summary JSON written by visualization/evaluate_experiments.py.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory containing *_eval_summary.json files.",
    )
    parser.add_argument(
        "--fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return a non-zero exit code if Notion sync is not configured or a request fails.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def repo_rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def discover_summary_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.summary_json:
        paths.extend(args.summary_json)
    if args.eval_dir:
        for eval_dir in args.eval_dir:
            paths.extend(sorted(eval_dir.glob("*_eval_summary.json")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def update_summary_sync_status(path: Path, status: dict[str, Any]) -> None:
    payload = load_json(path)
    payload["notion_sync"] = status
    write_json(path, payload)


class NotionConfig:
    def __init__(self) -> None:
        self.token = os.environ.get("NOTION_TOKEN") or os.environ.get("NOTION_API_KEY")
        self.data_source_id = os.environ.get("NOTION_EVAL_DATA_SOURCE_ID")
        self.database_id = os.environ.get("NOTION_EVAL_DATABASE_ID")
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
            return "missing dedicated NOTION_EVAL_DATA_SOURCE_ID or NOTION_EVAL_DATABASE_ID"
        return None


class NotionClient:
    def __init__(self, config: NotionConfig) -> None:
        self.config = config

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{NOTION_API_BASE}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Notion-Version": self.config.version,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Notion API {method} {path} failed with HTTP {exc.code}: {raw}") from exc
        if not raw:
            return {}
        return json.loads(raw)

    def retrieve_container(self) -> dict[str, Any]:
        if self.config.mode == "data_source":
            return self.request("GET", f"/data_sources/{self.config.container_id}")
        return self.request("GET", f"/databases/{self.config.container_id}")

    def query_container(self, filter_payload: dict[str, Any]) -> dict[str, Any]:
        payload = {"page_size": 1, "filter": filter_payload}
        if self.config.mode == "data_source":
            return self.request("POST", f"/data_sources/{self.config.container_id}/query", payload)
        return self.request("POST", f"/databases/{self.config.container_id}/query", payload)

    def create_page(self, properties: dict[str, Any], children: list[dict[str, Any]]) -> dict[str, Any]:
        parent_key = "data_source_id" if self.config.mode == "data_source" else "database_id"
        return self.request(
            "POST",
            "/pages",
            {
                "parent": {"type": parent_key, parent_key: self.config.container_id},
                "properties": properties,
                "children": children,
            },
        )

    def update_page(self, page_id: str, properties: dict[str, Any]) -> dict[str, Any]:
        return self.request("PATCH", f"/pages/{page_id}", {"properties": properties})


def plain_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def truncate_text(value: object, limit: int = MAX_RICH_TEXT) -> str:
    text = plain_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def property_type(schema: dict[str, Any]) -> str:
    return str(schema.get("type", ""))


def find_property(schema: dict[str, dict[str, Any]], aliases: tuple[str, ...], *, allowed: set[str] | None = None) -> str | None:
    lowered = {name.lower(): name for name in schema}
    for alias in aliases:
        name = lowered.get(alias.lower())
        if name and (allowed is None or property_type(schema[name]) in allowed):
            return name
    return None


def find_title_property(schema: dict[str, dict[str, Any]]) -> str:
    for name, prop in schema.items():
        if property_type(prop) == "title":
            return name
    raise ValueError("Notion database/data source has no title property")


def property_value(prop_schema: dict[str, Any], value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    kind = property_type(prop_schema)
    if kind == "title":
        return {"title": [{"text": {"content": truncate_text(value)}}]}
    if kind == "rich_text":
        return {"rich_text": [{"text": {"content": truncate_text(value)}}]}
    if kind == "number":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return {"number": number}
    if kind == "date":
        return {"date": {"start": truncate_text(value, 100)}}
    if kind == "select":
        text = truncate_text(value, 100)
        if not text:
            return None
        return {"select": {"name": text}}
    if kind == "multi_select":
        values = value if isinstance(value, list | tuple | set) else [value]
        return {"multi_select": [{"name": truncate_text(item, 100)} for item in values if plain_text(item)]}
    if kind == "url":
        text = plain_text(value)
        if text.startswith(("http://", "https://")):
            return {"url": text}
        return None
    if kind == "checkbox":
        return {"checkbox": bool(value)}
    return None


def add_if_present(
    *,
    properties: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    field: str,
    value: object,
    allowed: set[str] | None = None,
) -> None:
    prop_name = find_property(schema, PROPERTY_ALIASES[field], allowed=allowed)
    if not prop_name:
        return
    prop_value = property_value(schema[prop_name], value)
    if prop_value is not None:
        properties[prop_name] = prop_value


def summary_experiment_name(summary: dict[str, Any]) -> str:
    artifacts = summary.get("artifacts", {})
    experiment_dir = artifacts.get("experiment_dir")
    if experiment_dir:
        return Path(str(experiment_dir)).name
    checkpoint = summary.get("method", {}).get("checkpoint")
    if checkpoint:
        return Path(str(checkpoint)).parent.name
    return ""


def summary_parameterization(method: dict[str, Any]) -> str:
    for key in ("friction_parameterization", "parameterization"):
        if method.get(key):
            return str(method[key])
    return ""


def summary_sync_key(summary: dict[str, Any]) -> str:
    method = summary.get("method", {})
    return "|".join(
        [
            str(summary.get("eval_name", "")),
            str(summary.get("dataset", "")),
            str(method.get("checkpoint", "")),
            str(summary.get("max_steps", "")),
            ",".join(str(item) for item in summary.get("selected_trajectories", [])),
        ]
    )


def summary_title(summary: dict[str, Any]) -> str:
    experiment = summary_experiment_name(summary)
    method_name = summary.get("method", {}).get("name") or experiment
    eval_name = summary.get("eval_name", "eval")
    return f"{experiment} | {method_name} | {eval_name}"


def build_properties(summary: dict[str, Any], schema: dict[str, dict[str, Any]]) -> dict[str, Any]:
    method = summary.get("method", {})
    artifacts = summary.get("artifacts", {})
    properties: dict[str, Any] = {}
    title_property = find_title_property(schema)
    properties[title_property] = property_value(schema[title_property], summary_title(summary))

    add_if_present(properties=properties, schema=schema, field="sync_key", value=summary_sync_key(summary), allowed={"rich_text", "title"})
    add_if_present(
        properties=properties,
        schema=schema,
        field="time",
        value=summary.get("run_metadata", {}).get("created_at_utc") or utc_now_iso(),
        allowed={"date", "rich_text"},
    )
    add_if_present(properties=properties, schema=schema, field="status", value="Evaluated", allowed={"select", "rich_text"})
    add_if_present(properties=properties, schema=schema, field="experiment", value=summary_experiment_name(summary), allowed={"rich_text", "title"})
    add_if_present(properties=properties, schema=schema, field="method", value=method.get("name"), allowed={"rich_text", "title"})
    add_if_present(properties=properties, schema=schema, field="eval_name", value=summary.get("eval_name"), allowed={"rich_text", "select"})
    add_if_present(properties=properties, schema=schema, field="dataset", value=summary.get("dataset"), allowed={"rich_text", "url"})
    add_if_present(properties=properties, schema=schema, field="checkpoint", value=method.get("checkpoint"), allowed={"rich_text", "url"})
    add_if_present(properties=properties, schema=schema, field="checkpoint_type", value=method.get("checkpoint_type"), allowed={"select", "rich_text"})
    add_if_present(
        properties=properties,
        schema=schema,
        field="parameterization",
        value=summary_parameterization(method),
        allowed={"select", "rich_text"},
    )
    add_if_present(properties=properties, schema=schema, field="overlay_loss_mean", value=summary.get("overlay_loss_mean"), allowed={"number"})
    add_if_present(properties=properties, schema=schema, field="overlay_loss_std", value=summary.get("overlay_loss_std"), allowed={"number"})
    add_if_present(properties=properties, schema=schema, field="overlay_loss_min", value=summary.get("overlay_loss_min"), allowed={"number"})
    add_if_present(properties=properties, schema=schema, field="overlay_loss_max", value=summary.get("overlay_loss_max"), allowed={"number"})
    add_if_present(properties=properties, schema=schema, field="trajectory_count", value=summary.get("trajectory_count"), allowed={"number"})
    add_if_present(properties=properties, schema=schema, field="eval_dir", value=artifacts.get("eval_dir"), allowed={"rich_text", "url"})
    add_if_present(properties=properties, schema=schema, field="summary_path", value=artifacts.get("summary_json"), allowed={"rich_text", "url"})
    add_if_present(properties=properties, schema=schema, field="rollout_path", value=artifacts.get("trajectory_rollouts_json"), allowed={"rich_text", "url"})
    add_if_present(properties=properties, schema=schema, field="html_output_dir", value=artifacts.get("html_output_dir"), allowed={"rich_text", "url"})
    add_if_present(
        properties=properties,
        schema=schema,
        field="residual_gain",
        value=summary.get("pointnet_residual_gain"),
        allowed={"number", "rich_text"},
    )
    add_if_present(
        properties=properties,
        schema=schema,
        field="residual_output_mode",
        value=method.get("residual_output_mode") or summary.get("pointnet_residual_output_mode"),
        allowed={"select", "rich_text"},
    )
    return properties


def filter_for_existing_page(summary: dict[str, Any], schema: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sync_prop = find_property(schema, PROPERTY_ALIASES["sync_key"], allowed={"rich_text", "title"})
    if sync_prop:
        kind = property_type(schema[sync_prop])
        if kind == "title":
            return {"property": sync_prop, "title": {"equals": summary_sync_key(summary)}}
        return {"property": sync_prop, "rich_text": {"equals": summary_sync_key(summary)}}
    title_prop = find_title_property(schema)
    return {"property": title_prop, "title": {"equals": summary_title(summary)}}


def paragraph(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": truncate_text(text)}}]},
    }


def code_block(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "code",
        "code": {
            "language": "json",
            "rich_text": [{"type": "text", "text": {"content": truncate_text(text)}}],
        },
    }


def build_children(summary: dict[str, Any], summary_path: Path) -> list[dict[str, Any]]:
    method = summary.get("method", {})
    artifacts = summary.get("artifacts", {})
    compact = {
        "summary_json": repo_rel(summary_path),
        "dataset": summary.get("dataset"),
        "eval_name": summary.get("eval_name"),
        "max_steps": summary.get("max_steps"),
        "selected_trajectories": summary.get("selected_trajectories"),
        "loss_weights": summary.get("loss_weights"),
        "residual_gain": summary.get("pointnet_residual_gain"),
        "residual_output_mode": method.get("residual_output_mode") or summary.get("pointnet_residual_output_mode"),
        "method": method,
        "artifacts": artifacts,
    }
    json_text = json.dumps(compact, indent=2, sort_keys=True)
    chunks = [json_text[idx : idx + MAX_RICH_TEXT] for idx in range(0, len(json_text), MAX_RICH_TEXT)]
    children = [
        paragraph(f"Experiment: {summary_experiment_name(summary)}"),
        paragraph(f"Eval: {summary.get('eval_name')} | Dataset: {summary.get('dataset')}"),
        paragraph(
            "Overlay loss: "
            f"mean={summary.get('overlay_loss_mean')}, "
            f"min={summary.get('overlay_loss_min')}, "
            f"max={summary.get('overlay_loss_max')}"
        ),
    ]
    children.extend(code_block(chunk) for chunk in chunks[:4])
    return children


def sync_one_summary(client: NotionClient, schema: dict[str, dict[str, Any]], summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    properties = build_properties(summary, schema)
    query = client.query_container(filter_for_existing_page(summary, schema))
    results = query.get("results", [])
    if results:
        page = client.update_page(str(results[0]["id"]), properties)
        action = "updated"
    else:
        page = client.create_page(properties, build_children(summary, summary_path))
        action = "created"
    status = {
        "status": action,
        "synced_at_utc": utc_now_iso(),
        "page_id": page.get("id"),
        "page_url": page.get("url"),
        "container_mode": client.config.mode,
    }
    update_summary_sync_status(summary_path, status)
    return {"summary_json": repo_rel(summary_path), **status}


def skipped_status(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason, "synced_at_utc": utc_now_iso()}


def sync_eval_summaries_to_notion(summary_paths: list[Path], *, fail_on_error: bool = False) -> dict[str, Any]:
    config = NotionConfig()
    if not summary_paths:
        return {"status": "skipped", "reason": "no summary JSON files", "results": []}
    missing_reason = config.missing_reason
    if missing_reason is not None:
        status = skipped_status(missing_reason)
        for summary_path in summary_paths:
            update_summary_sync_status(summary_path, status)
        if fail_on_error:
            raise RuntimeError(missing_reason)
        return {"status": "skipped", "reason": missing_reason, "results": [{"summary_json": repo_rel(p), **status} for p in summary_paths]}

    client = NotionClient(config)
    try:
        container = client.retrieve_container()
        schema = container.get("properties", {})
        if not schema:
            raise ValueError("Notion database/data source returned no properties schema")
        results = [sync_one_summary(client, schema, summary_path) for summary_path in summary_paths]
        return {"status": "ok", "results": results}
    except Exception as exc:
        status = {"status": "error", "reason": str(exc), "synced_at_utc": utc_now_iso()}
        for summary_path in summary_paths:
            update_summary_sync_status(summary_path, status)
        if fail_on_error:
            raise
        return {"status": "error", "reason": str(exc), "results": [{"summary_json": repo_rel(p), **status} for p in summary_paths]}


def main() -> None:
    args = parse_args()
    summary_paths = discover_summary_paths(args)
    result = sync_eval_summaries_to_notion(summary_paths, fail_on_error=bool(args.fail_on_error))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
