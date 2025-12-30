#!/usr/bin/env python3
import os
import re
import csv
import argparse
from pathlib import Path


# Regex patterns (giữ nguyên)
RE_PDE_CLASS = re.compile(r"^PDE Class Name:\s*(.+)\s*$")
RE_VALIDATION = re.compile(
    r"Validation:\s*epoch\s+(\d+)\s+"
    r"MSE\s+([0-9eE+.\-]+)\s+"
    r"MAE\s+([0-9eE+.\-]+)\s+"
    r"MXE\s+([0-9eE+.\-]+)\s+"
    r"L1RE\s+([0-9eE+.\-]+)\s+"
    r"L2RE\s+([0-9eE+.\-]+)\s+"
    r"CRMSE\s+([0-9eE+.\-]+)"
    r"(?:\s+FRMSE\s+\(([0-9eE+.\-]+),\s*([0-9eE+.\-]+),\s*([0-9eE+.\-]+)\))?"
)


def parse_log(log_path: Path):
    # (giữ nguyên hoàn toàn như cũ)
    pde_class = None
    validation_epochs = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            m_pde = RE_PDE_CLASS.search(line)
            if m_pde:
                pde_class = m_pde.group(1)

            m_val = RE_VALIDATION.search(line)
            if m_val:
                epoch = int(m_val.group(1))
                metrics = {
                    "MSE": float(m_val.group(2)),
                    "MAE": float(m_val.group(3)),
                    "MXE": float(m_val.group(4)),
                    "L1RE": float(m_val.group(5)),
                    "L2RE": float(m_val.group(6)),
                    "CRMSE": float(m_val.group(7)),
                }
                if m_val.lastindex >= 10:
                    metrics["FRMSE_mean"] = float(m_val.group(8))
                    metrics["FRMSE_low"] = float(m_val.group(9))
                    metrics["FRMSE_high"] = float(m_val.group(10))
                else:
                    metrics["FRMSE_mean"] = ""
                    metrics["FRMSE_low"] = ""
                    metrics["FRMSE_high"] = ""

                validation_epochs[epoch] = metrics

    metrics_to_use = None
    if validation_epochs:
        final_epoch = max(validation_epochs.keys())
        metrics_to_use = validation_epochs[final_epoch]

    return {
        "pde_class": pde_class,
        "metrics": metrics_to_use,
    }


def infer_method_and_seed_from_run_dir(run_dir_name: str) -> tuple:
    # (giữ nguyên)
    if "-" in run_dir_name:
        suffix = run_dir_name.rsplit("-", 1)[-1]
        if "_" in suffix:
            method, seed = suffix.split("_", 1)
            return method, seed
        else:
            return suffix, ""
    return run_dir_name, ""


def collect_results(root: Path):
    # (giữ nguyên hoàn toàn)
    rows = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        if "log.txt" not in filenames:
            continue

        log_path = dirpath / "log.txt"

        run_dir = dirpath.parent.name
        method, seed = infer_method_and_seed_from_run_dir(run_dir)

        parsed = parse_log(log_path)

        pde_class = parsed["pde_class"]
        metrics = parsed["metrics"]

        row = {
            "method": method,
            "seed": seed,
            "pde_class": pde_class if pde_class is not None else "",
            "MSE": "",
            "MAE": "",
            "MXE": "",
            "L1RE": "",
            "L2RE": "",
            "CRMSE": "",
            "FRMSE_mean": "",
            "FRMSE_low": "",
            "FRMSE_high": "",
        }

        if metrics is not None:
            row.update({k: metrics.get(k, "") for k in row.keys() if k not in {"method", "seed", "pde_class"}})

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract final validation metrics (including FRMSE) from PINNacle log.txt files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="runs",
        help="Root directory containing run folders (default: runs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="Output CSV file (default: summary.csv)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    rows = collect_results(root)

    if not rows:
        print(f"No log.txt files found under {root}")
        return

    fieldnames = [
        "method",
        "seed",
        "pde_class",
        "MSE",
        "MAE",
        "MXE",
        "L1RE",
        "L2RE",
        "CRMSE",
        "FRMSE_mean",
        "FRMSE_low",
        "FRMSE_high",
    ]

    # Phần sửa: format float thành full thập phân (10 chữ số)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted_row = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted_row[k] = f"{v:.10f}".rstrip("0").rstrip(".")  # ví dụ 0.0007 thay vì 7e-4
                else:
                    formatted_row[k] = v if v != "" else ""
            writer.writerow(formatted_row)

    print(f"Wrote {len(rows)} rows to {args.output} (full decimal format)")


if __name__ == "__main__":
    main()