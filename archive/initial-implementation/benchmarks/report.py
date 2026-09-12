"""Render one run at a time, keeping environment and timing scopes separate."""

import argparse
import csv
import io
import json
from pathlib import Path


def rows(payload):
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark schema")
    records = payload["results"]
    baseline = {
        (r["kernel"], tuple(r["shape"]), r["dtype"], r["timing_mode"]): r["latency_us"]["median"]
        for r in records
        if r["status"] == "ok" and r["implementation"] == "pytorch"
    }
    result = []
    for r in records:
        key = (r["kernel"], tuple(r["shape"]), r["dtype"], r["timing_mode"])
        latency = r.get("latency_us", {})
        median = latency.get("median")
        result.append(
            {
                "kernel": r["kernel"],
                "shape": "x".join(map(str, r["shape"])),
                "dtype": r["dtype"],
                "implementation": r["implementation"],
                "timing_mode": r["timing_mode"],
                "status": r["status"],
                "p20_us": latency.get("p20"),
                "median_us": median,
                "p80_us": latency.get("p80"),
                "effective_GB_s": r.get("metrics", {}).get("effective_bandwidth_gbs"),
                "speedup_vs_pytorch": baseline[key] / median
                if key in baseline and median
                else None,
                "reason": r.get("reason", ""),
            }
        )
    return result


def markdown(payload):
    records = rows(payload)
    env = payload["environment"]
    hardware = env.get("hardware", {})
    lines = [
        "# GPU Performance Lab measurement report",
        "",
        f"Device: {hardware.get('gpu') or hardware.get('device') or 'unspecified'}  ",
        f"Commit: {env.get('git_commit')} (dirty: {env.get('git_dirty')})  ",
        f"Captured: {env.get('timestamp_utc')}",
        "",
        "Functional API timing with reused inputs. Effective GB/s uses minimum algorithmic",
        "traffic; it is not a DRAM counter. Speedups only match the same timing scope.",
        "",
        "| Kernel | Shape | Dtype | Implementation | Timing | Status | "
        "p20 / p50 / p80 µs | GB/s | Speedup |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in records:
        ok = row["status"] == "ok"
        times = (
            " / ".join(f"{row[k]:.3f}" for k in ("p20_us", "median_us", "p80_us")) if ok else "—"
        )
        bandwidth = f"{row['effective_GB_s']:.2f}" if ok else "—"
        speedup = f"{row['speedup_vs_pytorch']:.2f}x" if row["speedup_vs_pytorch"] else "—"
        lines.append(
            f"| {row['kernel']} | {row['shape']} | {row['dtype']} | {row['implementation']} | "
            f"{row['timing_mode']} | {row['status']} | {times} | {bandwidth} | {speedup} |"
        )
    if not any(r["status"] == "ok" for r in records):
        lines.extend(["", "Results pending hardware benchmark"])
    for row in records:
        if row["reason"]:
            reason = row["reason"].replace("\n", " ").replace("|", "\\|")
            lines.append(f"\n- {row['kernel']}/{row['implementation']} ({row['shape']}): {reason}")
    return "\n".join(lines) + "\n"


def render_csv(payload):
    data = rows(payload)
    out = io.StringIO()
    if data:
        writer = csv.DictWriter(out, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return out.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    payload = json.loads(args.input.read_text())
    content = markdown(payload) if args.format == "markdown" else render_csv(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
