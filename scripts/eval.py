#!/usr/bin/env python3
"""Reproduce the DRS VPT KITTI and nuScenes localization results with the released model.

    python scripts/eval.py --dataset kitti --data /data/kitti_vp2p_eval_data
    python scripts/eval.py --dataset nuscenes --data /data/nuscenes_test_processed/test
    python scripts/eval.py --dataset kitti --data ... --limit 20      # quick check on the first 20 samples

Samples run --batch-size at a time on the GPU while --workers processes load and voxelize the next batches; lower
--batch-size if the GPU runs out of memory.

Per-sample results go to <out>/<name>.json (an interrupted run resumes), and the summary to <out>/summary.json:
recall at RRE < 5 deg and RTE < 2 m, and the mean and median errors. See README.md for the data.
"""

import argparse
import json
from pathlib import Path

import torch

from drs_vpt.eval import SETS, evaluate_benchmark, summarize
from drs_vpt.hub import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(SETS))
    p.add_argument("--data", required=True, help="Dataset root (see README.md)")
    p.add_argument("--out", default=None, help="Output folder (default: eval_output/<dataset>)")
    p.add_argument("--checkpoint", default=None, help="Local save_pretrained dir or training .pth instead of the Hub model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    p.add_argument("--batch-size", type=int, default=8, help="Samples per forward pass (default: 8)")
    p.add_argument("--workers", type=int, default=8, help="Data loading processes (default: 8; 0 loads in-process)")
    args = p.parse_args()

    out = Path(args.out or f"eval_output/{args.dataset}")
    model = load_model(args.checkpoint, device=args.device)
    results = evaluate_benchmark(model, args.dataset, args.data, args.limit, out_dir=out,
                                 batch_size=args.batch_size, num_workers=args.workers)
    summary = {"dataset": args.dataset, **summarize(results)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
