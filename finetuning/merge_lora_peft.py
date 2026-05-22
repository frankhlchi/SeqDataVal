#!/usr/bin/env python
"""Merge PEFT LoRA adapter with base model and save as full HuggingFace model.

Usage:
    python merge_lora_peft.py \
        --adapter_path checkpoints/bipcov_mmlu \
        --base_model meta-llama/Llama-3.1-8B \
        --output_path checkpoints/bipcov_mmlu_merged
"""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(adapter_path: str, base_model: str, output_path: str):
    """Merge LoRA adapter with base model and save."""

    print(f"Loading tokenizer from: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU for merging
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("Done! Merged model saved successfully.")
    print(f"Output directory contents:")
    for f in Path(output_path).iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA with base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Base model name or path")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for merged model")
    args = parser.parse_args()

    merge_lora(args.adapter_path, args.base_model, args.output_path)


if __name__ == "__main__":
    main()
