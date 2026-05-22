#!/usr/bin/env python
"""Compute BM25 scores for DATE-LM finetune pool JSONL.

This is a lightweight replacement for DATE-LM's `methods/bm25/bm25_instruct.py` that
avoids the Lightning `InstructJsonDataModule` dependency, while keeping the same
high-level idea:
- Build a BM25 index over a small reference set (100 prompts from downstream eval)
- Score each train example by its average BM25 score against the reference set

Train JSONL rows are expected to contain a `messages` list.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm


def _messages_first_two(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    if messages[0].get("role") == "system":
        return messages[1:3]
    return messages[:2]


def _format_messages_tulu(messages: list[dict], eos_token: str) -> str:
    out = ""
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "system":
            out += "<|system|>\n" + content + "\n"
        elif role == "user":
            out += "<|user|>\n" + content + "\n"
        elif role == "assistant":
            out += "<|assistant|>\n" + content + eos_token + "\n"
        else:
            raise ValueError(f"Unsupported role: {role}")
    return out.strip()


def _iter_jsonl_messages(path: Path, start: int = 0) -> Iterator[list[dict]]:
    with path.open("r") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not isinstance(msgs, list):
                raise ValueError("Expected each JSONL row to have a `messages` list")
            yield msgs


def _count_lines(path: Path) -> int:
    with path.open("r") as f:
        return sum(1 for _ in f)


def _load_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {}
    try:
        return json.loads(progress_path.read_text())
    except Exception:
        return {}


def _save_progress(progress_path: Path, progress: dict) -> None:
    tmp = progress_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.replace(progress_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datelm_root", type=str, required=True)
    p.add_argument("--pool_jsonl", type=str, required=True)
    p.add_argument("--out_npy", type=str, required=True)

    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--task", type=str, required=True, choices=["mmlu", "gsm8k", "bbh"])
    p.add_argument("--ref_num_samples", type=int, default=100)
    p.add_argument("--ref_seed", type=int, default=42)

    p.add_argument(
        "--no_only_first_two",
        action="store_true",
        help="Disable DATE-LM `only_first_two=True` preprocessing for train messages.",
    )
    p.add_argument("--stopwords", type=str, default="en")
    p.add_argument("--k", type=int, default=1024)

    args = p.parse_args()

    datelm_root = Path(args.datelm_root)
    if not datelm_root.exists():
        raise FileNotFoundError(str(datelm_root))
    os.chdir(str(datelm_root))

    import bm25s
    from transformers import AutoTokenizer

    # DATE-LM dataset prompts (uses local `data/` via get_appropriate_data_dir)
    import sys

    sys.path.insert(0, str(datelm_root))
    from minimal_multitask.data import DATASETS

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Building reference set: task={args.task} n={args.ref_num_samples} seed={args.ref_seed}")
    val_ds = DATASETS[str(args.task)](tokenizer).get_all_test_prompts(
        num_samples=int(args.ref_num_samples),
        seed=int(args.ref_seed),
        prompt_only=False,
        response_only=False,
    )
    ref_texts = []
    for ex in val_ds:
        ids = ex["input_ids"]
        # ids is a torch tensor (dataset is set_format(torch))
        ref_texts.append(tokenizer.decode(ids.tolist(), skip_special_tokens=True))

    print(f"Indexing {len(ref_texts)} refs in BM25...")
    ref_tokens = bm25s.tokenize(ref_texts, stopwords=args.stopwords)
    retriever = bm25s.BM25()
    retriever.index(ref_tokens)

    pool_jsonl = Path(args.pool_jsonl)
    if not pool_jsonl.exists():
        raise FileNotFoundError(str(pool_jsonl))

    n_total = _count_lines(pool_jsonl)
    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    progress_path = out_npy.with_suffix(".progress.json")
    progress = _load_progress(progress_path)

    if out_npy.exists() and progress.get("complete") is True:
        print(f"Scores already complete: {out_npy}")
        return

    start_idx = int(progress.get("next_idx", 0))
    if start_idx < 0 or start_idx > n_total:
        start_idx = 0

    mode = "r+" if out_npy.exists() else "w+"
    scores_mm = open_memmap(out_npy, mode=mode, dtype=np.float32, shape=(n_total,))

    only_first_two = not bool(args.no_only_first_two)
    iterator = _iter_jsonl_messages(pool_jsonl, start=start_idx)

    write_idx = start_idx
    for msgs in tqdm(iterator, total=(n_total - start_idx), desc="BM25 scoring"):
        if only_first_two:
            msgs = _messages_first_two(msgs)
        text = _format_messages_tulu(msgs, eos_token=tokenizer.eos_token)
        tok = bm25s.tokenize([text], stopwords=args.stopwords)
        result = retriever.retrieve(tok, k=min(int(args.k), len(ref_texts)))
        scores_mm[write_idx] = float(result.scores.mean())
        write_idx += 1

        if write_idx % 2048 == 0:
            progress["next_idx"] = write_idx
            _save_progress(progress_path, progress)

    progress["next_idx"] = write_idx
    progress["complete"] = True
    _save_progress(progress_path, progress)

    print(f"Saved scores -> {out_npy} (rows={write_idx})")


if __name__ == "__main__":
    main()
