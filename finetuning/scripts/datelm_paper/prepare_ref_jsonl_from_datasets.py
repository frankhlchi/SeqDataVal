#!/usr/bin/env python
"""Generate a DATE-LM `minimal_multitask.data.DATASETS`-based reference JSONL.

Motivation
----------
DATE-LM baselines (RepSim/RDS+/BM25 in this repo) build their reference set via
`minimal_multitask.data.DATASETS[task].get_all_test_prompts(...)`, typically with:
  - prompt_only=False
  - response_only=False
which includes both the prompt and the label (benchmark protocol; label leakage is
inherent in DATE-LM's setup).

For fair comparisons, we generate the *same* ref samples and materialize them as
a JSONL with a single `text` field, so non-LLM embedding methods (e.g., BipCov with
SentenceTransformer embeddings) can reuse it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datelm_root",
        type=str,
        required=True,
        help="Path to DATE-LM root (repo root or DATE-LM-main).",
    )
    p.add_argument("--task", type=str, required=True, choices=["mmlu", "gsm8k", "bbh"])
    p.add_argument("--out_jsonl", type=str, required=True)

    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt_only", action="store_true", help="Generate prompt-only refs (no label).")
    p.add_argument("--response_only", action="store_true", help="Generate label-only refs.")
    p.add_argument(
        "--keep_special_tokens",
        action="store_true",
        help="Keep special tokens when decoding input_ids back to text.",
    )
    args = p.parse_args()

    datelm_root = Path(args.datelm_root)
    if not datelm_root.exists():
        raise FileNotFoundError(str(datelm_root))

    # DATASETS expects running from DATE-LM root so it can find `data/`.
    os.chdir(str(datelm_root))
    sys.path.insert(0, str(datelm_root))

    from transformers import AutoTokenizer

    from minimal_multitask.data import DATASETS

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = DATASETS[str(args.task)](tokenizer).get_all_test_prompts(
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        prompt_only=bool(args.prompt_only),
        response_only=bool(args.response_only),
    )

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_jsonl.with_suffix(".meta.json")

    skip_special_tokens = not bool(args.keep_special_tokens)
    written = 0
    with out_jsonl.open("w") as f:
        for ex in ds:
            ids = ex["input_ids"]
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=skip_special_tokens)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "task": str(args.task),
        "model_name": str(args.model_name),
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "prompt_only": bool(args.prompt_only),
        "response_only": bool(args.response_only),
        "skip_special_tokens": bool(skip_special_tokens),
        "out_jsonl": str(out_jsonl),
        "rows": written,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"Wrote {written} refs -> {out_jsonl}")
    print(f"Wrote meta -> {meta_path}")


if __name__ == "__main__":
    main()
