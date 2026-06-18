from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "eval" / "historical_experiment_registry" / "registry.json"
DEFAULT_REPORT_DIR = ROOT / "eval" / "unified_historical_eval"
PROTOCOLS = {
    "rotation68": {
        "dataset": ROOT
        / "mujoco/outputs/rotation_friction_diagnostics_l0p20_r0p50_68"
        / "same_mean_split_left_0p20_right_0p50/same_mean_split_left_0p20_right_0p50.npz",
        "max_steps": "300",
        "eval_batch_size": "20",
        "trajectory_count": 68,
    },
    "very_long20": {
        "dataset": ROOT
        / "mujoco/outputs/very_long_rotation_friction_diagnostics_l0p20_r0p50_20"
        / "same_mean_split_left_0p20_right_0p50/same_mean_split_left_0p20_right_0p50.npz",
        "max_steps": "full",
        "eval_batch_size": "4",
        "trajectory_count": 20,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--registry-json", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), action="append", default=None)
    parser.add_argument("--family", type=str, action="append", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_registry(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a records list")
    return records


def matching_existing_summary(checkpoint: Path, protocol_name: str) -> Path | None:
    eval_dir = checkpoint.parent / "eval" / protocol_name
    for summary_path in sorted(eval_dir.glob("*_eval_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        method = summary.get("method") if isinstance(summary.get("method"), dict) else {}
        summary_checkpoint = resolve_repo_path(str(method.get("checkpoint") or ""))
        if (
            summary.get("schema_version") == "experiment-eval-v3"
            and summary.get("metric_version") == "trajectory-fit-v1"
            and summary.get("eval_name") == protocol_name
            and summary_checkpoint.resolve() == checkpoint.resolve()
            and summary.get("trajectory_count") == PROTOCOLS[protocol_name]["trajectory_count"]
            and summary.get("max_steps")
            == (None if PROTOCOLS[protocol_name]["max_steps"] == "full" else int(PROTOCOLS[protocol_name]["max_steps"]))
        ):
            return summary_path
    return None


def eval_command(checkpoint: Path, protocol_name: str, device: str, family: str) -> list[str]:
    protocol = PROTOCOLS[protocol_name]
    if family == "contact_field":
        return [
            sys.executable,
            str(ROOT / "visualization" / "evaluate_contact_field_unified.py"),
            "--dataset",
            str(protocol["dataset"]),
            "--eval-name",
            protocol_name,
            "--checkpoint",
            str(checkpoint),
            "--device",
            str(device),
            "--eval-batch-size",
            str(protocol["eval_batch_size"]),
            "--max-steps",
            str(protocol["max_steps"]),
        ]
    return [
        sys.executable,
        str(ROOT / "visualization" / "evaluate_experiments.py"),
        "--dataset",
        str(protocol["dataset"]),
        "--eval-name",
        protocol_name,
        "--checkpoint",
        str(checkpoint),
        "--device",
        str(device),
        "--eval-batch-size",
        str(protocol["eval_batch_size"]),
        "--max-steps",
        str(protocol["max_steps"]),
        "--no-sync-notion",
    ]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocols = list(dict.fromkeys(args.protocol or list(PROTOCOLS)))
    requested_families = set(args.family or [])
    records = load_registry(args.registry_json)
    if requested_families:
        records = [record for record in records if str(record.get("family")) in requested_families]
    if args.limit is not None:
        records = records[: max(int(args.limit), 0)]

    args.report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.report_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "report.json"
    report: dict[str, Any] = {
        "schema_version": "unified-historical-eval-v1",
        "started_at_utc": utc_now_iso(),
        "registry_json": str(args.registry_json),
        "protocols": protocols,
        "device": args.device,
        "record_count": len(records),
        "results": [],
    }
    write_report(report_path, report)

    total_jobs = len(records) * len(protocols)
    job_idx = 0
    for record in records:
        family = str(record.get("family") or "")
        checkpoint = resolve_repo_path(str(record.get("artifact_path") or ""))
        for protocol_name in protocols:
            job_idx += 1
            result: dict[str, Any] = {
                "record_id": record.get("record_id"),
                "experiment_name": record.get("experiment_name"),
                "family": family,
                "checkpoint": str(checkpoint),
                "protocol": protocol_name,
                "started_at_utc": utc_now_iso(),
            }
            if not checkpoint.exists():
                result.update(status="failed", reason="checkpoint does not exist")
                report["results"].append(result)
                write_report(report_path, report)
                print(f"[{job_idx}/{total_jobs}] missing {checkpoint} {protocol_name}", flush=True)
                if args.fail_fast:
                    raise FileNotFoundError(checkpoint)
                continue
            existing = matching_existing_summary(checkpoint, protocol_name) if args.skip_existing else None
            if existing is not None:
                result.update(status="skipped_existing", summary_json=str(existing))
                report["results"].append(result)
                write_report(report_path, report)
                print(f"[{job_idx}/{total_jobs}] existing {checkpoint.name} {protocol_name}", flush=True)
                continue

            log_path = logs_dir / f"{job_idx:04d}_{checkpoint.stem}_{protocol_name}.log"
            command = eval_command(checkpoint, protocol_name, args.device, family)
            start = time.time()
            print(f"[{job_idx}/{total_jobs}] running {family} {checkpoint.name} {protocol_name}", flush=True)
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                log_path.write_text(completed.stdout, encoding="utf-8")
                elapsed = float(time.time() - start)
                summary = matching_existing_summary(checkpoint, protocol_name)
                if completed.returncode == 0 and summary is not None:
                    result.update(
                        status="completed",
                        elapsed_seconds=elapsed,
                        summary_json=str(summary),
                        log=str(log_path),
                    )
                    print(f"[{job_idx}/{total_jobs}] completed in {elapsed:.1f}s", flush=True)
                else:
                    result.update(
                        status="failed",
                        elapsed_seconds=elapsed,
                        returncode=int(completed.returncode),
                        reason="eval command failed or did not produce a matching v3 summary",
                        log=str(log_path),
                    )
                    print(f"[{job_idx}/{total_jobs}] failed rc={completed.returncode} log={log_path}", flush=True)
                    if args.fail_fast:
                        raise RuntimeError(f"eval failed: {checkpoint} {protocol_name}")
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                log_path.write_text(str(output), encoding="utf-8")
                result.update(
                    status="failed",
                    elapsed_seconds=float(time.time() - start),
                    reason=f"timeout after {args.timeout_seconds} seconds",
                    log=str(log_path),
                )
                print(f"[{job_idx}/{total_jobs}] timeout log={log_path}", flush=True)
                if args.fail_fast:
                    raise
            report["results"].append(result)
            write_report(report_path, report)

    report["finished_at_utc"] = utc_now_iso()
    counts: dict[str, int] = {}
    for result in report["results"]:
        status = str(result.get("status"))
        counts[status] = counts.get(status, 0) + 1
    report["status_counts"] = counts
    write_report(report_path, report)
    print(json.dumps({"report": str(report_path), "status_counts": counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
