#!/usr/bin/env python
"""Compute embeddings for DATE-LM fine-tuning data selection.

This script produces:
  - train_emb.npy: embeddings for training pool (e.g., Tulu3)
  - ref_emb.npy: embeddings for reference/validation set (e.g., MMLU prompts)

Usage:
    # From JSONL files
    python compute_embeddings_for_datelm.py \
        --train_jsonl /path/to/tulu3.jsonl \
        --ref_jsonl /path/to/mmlu_prompts.jsonl \
        --out_dir embeddings/mmlu \
        --model BAAI/bge-large-en-v1.5 \
        --device cuda

    # From HuggingFace datasets
    python compute_embeddings_for_datelm.py \
        --train_hf allenai/tulu-3-sft-mixture \
        --ref_hf cais/mmlu \
        --ref_hf_subset all \
        --out_dir embeddings/mmlu \
        --model BAAI/bge-large-en-v1.5 \
        --device cuda

    # Using LLM hidden states (slower but may align better with fine-tuning)
    python compute_embeddings_for_datelm.py \
        --train_jsonl /path/to/tulu3.jsonl \
        --ref_jsonl /path/to/mmlu_prompts.jsonl \
        --out_dir embeddings/mmlu_llm \
        --use_llm \
        --llm_model meta-llama/Llama-3.1-8B \
        --device cuda

    # Compute only train embeddings (reuse across tasks)
    python compute_embeddings_for_datelm.py \
        --train_jsonl /path/to/tulu3_200k.jsonl \
        --out_dir embeddings/tulu3_200k_bge \
        --no_ref \
        --model BAAI/bge-large-en-v1.5 \
        --device cuda

    # Compute only ref embeddings (reuse an existing train_emb.npy)
    python compute_embeddings_for_datelm.py \
        --ref_hf cais/mmlu --ref_hf_subset all --ref_hf_split test --ref_max_samples 100 \
        --out_dir embeddings/mmlu_ref_bge \
        --no_train \
        --model BAAI/bge-large-en-v1.5 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm


def _messages_first_two(messages: list[dict]) -> list[dict]:
    """Mirror DATE-LM `only_first_two` behavior for multi-turn chat."""
    if not messages:
        return messages
    if messages[0].get("role") == "system":
        return messages[1:3]
    return messages[:2]


def load_texts_from_jsonl(
    path: str,
    max_samples: int = -1,
    text_field: str = "auto",
    messages_only_first_two: bool = False,
) -> List[str]:
    """Load texts from JSONL file."""
    texts: List[str] = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            item = json.loads(line)

            # Auto-detect text field
            if text_field == "auto":
                if "messages" in item:
                    messages = item["messages"]
                    if messages_only_first_two:
                        messages = _messages_first_two(messages)
                    # Chat format: concatenate selected messages
                    text = " ".join(m.get("content", "") for m in messages if m.get("content"))
                elif "text" in item:
                    text = item["text"]
                elif "input" in item:
                    text = item["input"]
                    if "output" in item:
                        text += " " + item["output"]
                elif "question" in item:
                    text = item["question"]
                elif "prompt" in item:
                    text = item["prompt"]
                else:
                    text = str(item)
            else:
                text = item.get(text_field, str(item))

            texts.append(text.strip())

    print(f"Loaded {len(texts)} texts from {path}")
    return texts


def load_texts_from_hf(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    max_samples: int = -1,
    seed: int = 42,
    messages_only_first_two: bool = False,
) -> List[str]:
    """Load texts from HuggingFace dataset."""
    from datasets import load_dataset

    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    # Sample if needed
    if max_samples > 0 and len(ds) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), size=max_samples, replace=False)
        ds = ds.select(indices)

    texts: List[str] = []
    for item in ds:
        # Try common field names
        if "messages" in item and item["messages"]:
            messages = item["messages"]
            if messages_only_first_two:
                messages = _messages_first_two(messages)
            text = " ".join(m.get("content", "") for m in messages if m.get("content"))
        elif "text" in item:
            text = item["text"]
        elif "question" in item:
            text = item["question"]
        elif "input" in item:
            text = item["input"]
        elif "prompt" in item:
            text = item["prompt"]
        else:
            text = str(item)
        texts.append(text.strip())

    print(f"Loaded {len(texts)} texts from {dataset_name}")
    return texts


def compute_embeddings_st(
    texts: List[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cuda",
    batch_size: int = 64,
    normalize: bool = True,
    max_length: int = 512,
) -> np.ndarray:
    """Compute embeddings using SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Truncate long texts
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def compute_embeddings_llm(
    texts: List[str],
    model_name: str = "meta-llama/Llama-3.1-8B",
    device: str = "cuda",
    batch_size: int = 4,
    max_length: int = 2048,
) -> np.ndarray:
    """Compute embeddings using LLM hidden states (last token)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    all_embeddings = []
    print(f"Computing LLM embeddings for {len(texts)} texts...")

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get last layer hidden states
            hidden_states = outputs.hidden_states[-1]
            # Get last non-padding token representation
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            batch_size_actual = hidden_states.size(0)
            last_token_reps = hidden_states[
                torch.arange(batch_size_actual, device=device), seq_lengths
            ]
            all_embeddings.append(last_token_reps.cpu().float().numpy())

    embeddings = np.vstack(all_embeddings)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    return embeddings.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()

    # Input options (train)
    parser.add_argument("--train_jsonl", type=str, help="Path to training JSONL")
    parser.add_argument("--train_hf", type=str, help="HuggingFace dataset for training")
    parser.add_argument("--train_hf_subset", type=str, help="HF dataset subset")
    parser.add_argument("--train_hf_split", type=str, default="train")
    parser.add_argument("--train_max_samples", type=int, default=200000)

    # Input options (ref)
    parser.add_argument("--ref_jsonl", type=str, help="Path to reference JSONL")
    parser.add_argument("--ref_hf", type=str, help="HuggingFace dataset for reference")
    parser.add_argument("--ref_hf_subset", type=str, help="HF dataset subset for ref")
    parser.add_argument("--ref_hf_split", type=str, default="test")
    parser.add_argument("--ref_max_samples", type=int, default=100)

    # Output
    parser.add_argument("--out_dir", type=str, required=True)

    # What to compute
    parser.add_argument(
        "--no_train",
        action="store_true",
        help="Skip computing train_emb.npy (only compute ref embeddings).",
    )
    parser.add_argument(
        "--no_ref",
        action="store_true",
        help="Skip computing ref_emb.npy (only compute train embeddings).",
    )

    # Data formatting
    parser.add_argument(
        "--messages_only_first_two",
        action="store_true",
        help=(
            "When reading JSONL/HF items with `messages`, only use the first user+assistant turn "
            "(matches DATE-LM instruct preprocessing with `only_first_two=True`)."
        ),
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        help="Use LLM hidden states instead of SentenceTransformer",
    )
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.1-8B")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_emb = None
    ref_emb = None

    # Load + embed train texts
    if not args.no_train:
        if args.train_jsonl:
            train_texts = load_texts_from_jsonl(
                args.train_jsonl,
                args.train_max_samples,
                messages_only_first_two=args.messages_only_first_two,
            )
        elif args.train_hf:
            train_texts = load_texts_from_hf(
                args.train_hf,
                args.train_hf_subset,
                args.train_hf_split,
                args.train_max_samples,
                args.seed,
                messages_only_first_two=args.messages_only_first_two,
            )
        else:
            raise ValueError("Provide --train_jsonl or --train_hf (or pass --no_train).")

        if args.use_llm:
            train_emb = compute_embeddings_llm(
                train_texts,
                args.llm_model,
                args.device,
                batch_size=min(args.batch_size, 8),
                max_length=args.max_length,
            )
        else:
            train_emb = compute_embeddings_st(
                train_texts,
                args.model,
                args.device,
                args.batch_size,
                normalize=True,
                max_length=args.max_length,
            )

        np.save(out_dir / "train_emb.npy", train_emb)
        print(f"\nSaved:")
        print(f"  {out_dir / 'train_emb.npy'}: {train_emb.shape}")

    # Load + embed ref texts
    if not args.no_ref:
        if args.ref_jsonl:
            ref_texts = load_texts_from_jsonl(
                args.ref_jsonl,
                args.ref_max_samples,
                messages_only_first_two=args.messages_only_first_two,
            )
        elif args.ref_hf:
            ref_texts = load_texts_from_hf(
                args.ref_hf,
                args.ref_hf_subset,
                args.ref_hf_split,
                args.ref_max_samples,
                args.seed,
                messages_only_first_two=args.messages_only_first_two,
            )
        else:
            raise ValueError("Provide --ref_jsonl or --ref_hf (or pass --no_ref).")

        if args.use_llm:
            ref_emb = compute_embeddings_llm(
                ref_texts,
                args.llm_model,
                args.device,
                batch_size=min(args.batch_size, 8),
                max_length=args.max_length,
            )
        else:
            ref_emb = compute_embeddings_st(
                ref_texts,
                args.model,
                args.device,
                args.batch_size,
                normalize=True,
                max_length=args.max_length,
            )

        np.save(out_dir / "ref_emb.npy", ref_emb)
        print(f"\nSaved:")
        print(f"  {out_dir / 'ref_emb.npy'}: {ref_emb.shape}")

    meta = {
        "train_shape": list(train_emb.shape) if train_emb is not None else None,
        "ref_shape": list(ref_emb.shape) if ref_emb is not None else None,
        "model": args.llm_model if args.use_llm else args.model,
        "use_llm": args.use_llm,
        "messages_only_first_two": args.messages_only_first_two,
        "train_source": None if args.no_train else (args.train_jsonl or args.train_hf),
        "ref_source": None if args.no_ref else (args.ref_jsonl or args.ref_hf),
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
