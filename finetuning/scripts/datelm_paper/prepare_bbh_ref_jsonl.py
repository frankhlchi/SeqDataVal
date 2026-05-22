#!/usr/bin/env python
"""Create a small BBH reference JSONL by sampling inputs from local DATE-LM eval data.

We sample across tasks under:
  $DATELM_ROOT/data/eval/bbh/bbh/*.json

Each task JSON has an `examples` list with fields `input` and `target`.

Example:
  python prepare_bbh_ref_jsonl.py \
    --bbh_dir "$DATELM_ROOT/data/eval/bbh/bbh" \
    --out_jsonl "$DATELM_ROOT/data/eval/bbh/paper_seed42_v1_bbh_ref_100.jsonl" \
    --k 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bbh_dir", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import random

    bbh_dir = Path(args.bbh_dir)
    if not bbh_dir.exists():
        raise FileNotFoundError(str(bbh_dir))

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    tasks = sorted(bbh_dir.glob("*.json"))
    if not tasks:
        raise RuntimeError(f"No *.json found under {bbh_dir}")

    rows = []
    for task_path in tasks:
        task_name = task_path.stem
        data = json.loads(task_path.read_text())
        examples = data.get("examples", [])
        for ex in examples:
            inp = ex.get("input")
            if not inp:
                continue
            rows.append({"task": task_name, "input": inp})

    if not rows:
        raise RuntimeError("No inputs extracted from BBH JSON files")

    rng = random.Random(int(args.seed))
    k = int(min(args.k, len(rows)))
    sampled = rng.sample(rows, k)

    with out_jsonl.open("w") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {k} BBH ref examples -> {out_jsonl}")


if __name__ == "__main__":
    main()
