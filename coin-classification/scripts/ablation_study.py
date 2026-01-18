"""Run an ablation study across multiple CLI configurations.

This script launches the unified CLI with multiple parameter combinations and
stores a summary CSV of metrics found in each run directory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _build_cmd(base_cmd: List[str], params: Dict[str, str]) -> List[str]:
    cmd = list(base_cmd)
    for k, v in params.items():
        if v is None:
            continue
        cmd.extend([k, str(v)])
    return cmd


def _collect_metrics(run_dir: Path) -> Dict:
    metrics = {"run_dir": str(run_dir)}

    metric_files = sorted(run_dir.glob("metrics_epoch_*.json"))
    if metric_files:
        latest = metric_files[-1]
        data = json.loads(latest.read_text(encoding="utf-8"))
        metrics.update({
            "val_accuracy": data.get("accuracy"),
            "val_f1_weighted": data.get("f1_weighted"),
            "val_loss": data.get("val_loss"),
        })

    knn_file = run_dir / "knn_metrics.json"
    if knn_file.exists():
        data = json.loads(knn_file.read_text(encoding="utf-8"))
        metrics.update({
            "knn_accuracy": data.get("accuracy"),
            "knn_f1_weighted": data.get("f1_weighted"),
        })

    hdbscan_file = run_dir / "hdbscan_metrics.json"
    if hdbscan_file.exists():
        data = json.loads(hdbscan_file.read_text(encoding="utf-8"))
        metrics.update({
            "hdbscan_n_clusters": data.get("n_clusters"),
            "hdbscan_noise_pct": data.get("noise_pct"),
            "hdbscan_silhouette": data.get("silhouette"),
        })

    return metrics


def main():
    # python ./classification/scripts/ablation_study.py --data "D:\Studium\Master\sem_3_wise_2025\DC\muenz_projekt\merged_dataset_pairs.tsv" --out-root ./classification/runs/ablation_2026_01_18_17_12 --mode classifier
    # python ./classification/scripts/ablation_study.py --data "D:\Studium\Master\sem_3_wise_2025\DC\muenz_projekt\merged_dataset_pairs.tsv" --out-root ./classification/runs/ablation_2026_01_18_17_51 --mode metric --eval-method knn

    parser = argparse.ArgumentParser(description="Run ablation study for coin classification")
    parser.add_argument("--data", required=True, help="Path to TSV/CSV dataset")
    parser.add_argument("--out-root", default="runs/ablation", help="Root output directory")
    parser.add_argument("--mode", default="classifier", choices=["classifier", "metric"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--views", default="both_concat", help="Comma-separated views list")
    parser.add_argument("--head-types", default="linear,mlp", help="Comma-separated classifier head types")
    parser.add_argument("--proj-types", default="linear,mlp", help="Comma-separated metric proj types")
    parser.add_argument("--embedding-dims", default="256,512", help="Comma-separated embedding dims")
    parser.add_argument("--eval-method", default="none", choices=["none", "knn", "hdbscan"])
    parser.add_argument("--min-condition", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only print commands")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    views_list = [v.strip() for v in args.views.split(",") if v.strip()]
    head_types = [h.strip() for h in args.head_types.split(",") if h.strip()]
    proj_types = [p.strip() for p in args.proj_types.split(",") if p.strip()]
    embedding_dims = [int(x.strip()) for x in args.embedding_dims.split(",") if x.strip()]

    base_cmd = [sys.executable, "-m", "coin_classifier.cli", "--data", args.data,
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--lr", str(args.lr),
                "--mode", args.mode, "--eval-method", args.eval_method, "--wandb"]

    if args.min_condition is not None:
        base_cmd.extend(["--min-condition", str(args.min_condition)])

    runs = []

    if args.mode == "classifier":
        grid = itertools.product(views_list, head_types)
        for views, head_type in grid:
            run_name = f"cls_{views}_{head_type}"
            run_dir = out_root / run_name
            params = {
                "--out-dir": str(run_dir),
                "--views": views,
                "--head-type": head_type,
            }
            cmd = _build_cmd(base_cmd, params)
            runs.append((cmd, run_dir))
    else:
        grid = itertools.product(views_list, proj_types, embedding_dims)
        for views, proj_type, emb_dim in grid:
            run_name = f"metric_{views}_{proj_type}_{emb_dim}"
            run_dir = out_root / run_name
            params = {
                "--out-dir": str(run_dir),
                "--views": views,
                "--proj-type": proj_type,
                "--embedding-dim": emb_dim,
            }
            cmd = _build_cmd(base_cmd, params)
            runs.append((cmd, run_dir))

    for cmd, run_dir in runs:
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
