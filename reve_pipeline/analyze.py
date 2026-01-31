#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze.py
- Scans results/<TAG>/run_*/run_meta.json
- For each TAG, picks latest run (by folder name) unless RUN_ID is specified
- Produces summary_table.csv under results/
"""

import os
import json
import glob
import csv
from typing import Dict, List, Optional

REPO_ROOT = r"D:\REPOS\alz-ftd-ctl-reve"
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")

TARGET_TAGS = ["A-F", "A-C", "F-C", "A-F-C"]


def load_json(p: str) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_latest_run_dir(tag_dir: str) -> Optional[str]:
    runs = sorted(glob.glob(os.path.join(tag_dir, "run_*")))
    if not runs:
        return None
    return runs[-1]


def main():
    summary_rows: List[Dict] = []

    for tag in TARGET_TAGS:
        tag_dir = os.path.join(RESULTS_ROOT, tag)
        run_dir = pick_latest_run_dir(tag_dir)
        if run_dir is None:
            print(f"[WARN] No runs found for {tag} under {tag_dir}")
            continue

        meta_path = os.path.join(run_dir, "run_meta.json")
        if not os.path.exists(meta_path):
            print(f"[WARN] Missing run_meta.json for {tag}: {run_dir}")
            continue

        obj = load_json(meta_path)
        meta = obj.get("meta", {})
        g = obj.get("global_metrics", {})

        summary_rows.append({
            "tag": tag,
            "run_id": meta.get("run_id", os.path.basename(run_dir)),
            "created_at": meta.get("created_at", ""),
            "n_subjects": meta.get("n_subjects", ""),
            "class_order_used": ",".join(meta.get("class_order_used", [])),
            "acc": g.get("acc", ""),
            "bal_acc": g.get("bal_acc", ""),
            "mcc": g.get("mcc", ""),
            "macro_f1": g.get("macro_f1", ""),
            "weighted_f1": g.get("weighted_f1", ""),
            "sens": g.get("sens", ""),  # binary only
            "spec": g.get("spec", ""),  # binary only
            "run_dir": run_dir,
        })

        print(f"[{tag}] Using run: {run_dir}")

    if not summary_rows:
        print("No summaries produced.")
        return

    out_csv = os.path.join(RESULTS_ROOT, "summary_table.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 72)
    print("Saved:", out_csv)
    print("=" * 72)


if __name__ == "__main__":
    main()
