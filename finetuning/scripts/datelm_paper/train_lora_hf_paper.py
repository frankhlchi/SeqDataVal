#!/usr/bin/env python
"""LoRA fine-tuning (HF+PEFT) approximating DATE-LM / paper settings.

Key alignments:
  - Base model: meta-llama/Llama-3.1-8B (base)
  - epochs=2, lr=2e-5, effective batch=128 (default: bsz=4, grad_accum=32)
  - Prompt masking via DATE-LM `encode_with_messages_format` (assistant-only loss)
  - `only_first_two=True` (matches DATE-LM InstructJsonDataModule)
  - LoRA target modules: q_proj/k_proj/v_proj/o_proj (no MLP)

Note on dataloader shuffling:
  - By default, HF Trainer uses a shuffled sampler for training.
  - DATE-LM's LitGPT dataloader uses `shuffle=False`.
  - Use `--no_shuffle_train` if you want to match DATE-LM's no-shuffle behavior.

Example:
  python train_lora_hf_paper.py \
    --datelm_root /path/to/DATE-LM \
    --train_jsonl /path/to/DATE-LM/selected_data/<TAG>/mmlu/random_10k.jsonl \
    --output_dir /path/to/DATE-LM/checkpoints/<TAG>_random_mmlu_lora \
    --model_name meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


@dataclass
class DataCollatorCausalLM:
    tokenizer: Any
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        max_len = batch["input_ids"].shape[1]
        labels_padded: List[torch.Tensor] = []
        for l in labels:
            l = torch.tensor(l, dtype=torch.long)
            if l.shape[0] < max_len:
                pad = torch.full((max_len - l.shape[0],), self.label_pad_token_id, dtype=torch.long)
                l = torch.cat([l, pad], dim=0)
            labels_padded.append(l)
        batch["labels"] = torch.stack(labels_padded, dim=0)
        return batch


class NoShuffleTrainer(Trainer):
    """Match DATE-LM dataloader behavior (shuffle=False) for determinism."""

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")

        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datelm_root",
        type=str,
        required=True,
        help="Path to DATE-LM root (repo root or DATE-LM-main), for imports/data lookup.",
    )
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--max_seq_length", type=int, default=2048)

    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=512)
    p.add_argument("--lora_dropout", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--max_samples", type=int, default=-1, help="Optional cap on number of training samples")
    p.add_argument(
        "--save_optimizer_bin",
        action="store_true",
        help="Save HF optimizer.state_dict() to <output_dir>/optimizer.bin (needed by DATE-LM LESS scorer).",
    )
    p.add_argument(
        "--no_shuffle_train",
        action="store_true",
        help="Disable shuffling in the training dataloader (match DATE-LM shuffle=False).",
    )
    args = p.parse_args()

    datelm_root = Path(args.datelm_root)
    if not datelm_root.exists():
        raise FileNotFoundError(str(datelm_root))
    sys.path.insert(0, str(datelm_root))

    from minimal_multitask.utils import encode_with_messages_format

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading train data: {args.train_jsonl}")
    ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Train rows: {len(ds)}")

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        encoded = encode_with_messages_format(
            example,
            tokenizer,
            args.max_seq_length,
            True,   # include_response
            False,  # response_only
            True,   # only_first_two
            False,  # prompt_only
            False,  # add_bos_token
        )
        return {
            "input_ids": encoded["input_ids"].tolist(),
            "attention_mask": encoded["attention_mask"].tolist(),
            "labels": encoded["labels"].tolist(),
        }

    print("Tokenizing (DATE-LM messages format; assistant-only loss)...")
    ds = ds.map(preprocess, remove_columns=ds.column_names)

    collator = DataCollatorCausalLM(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
    )

    trainer_cls = NoShuffleTrainer if args.no_shuffle_train else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.save_optimizer_bin:
        if trainer.optimizer is None:
            raise RuntimeError("Trainer optimizer is None; cannot save optimizer.bin.")
        opt_path = output_dir / "optimizer.bin"
        print(f"Saving optimizer state -> {opt_path} (can be large for LoRA r={args.lora_r})")
        torch.save(trainer.optimizer.state_dict(), opt_path)

    cfg = {
        "datelm_root": str(datelm_root),
        "base_model": args.model_name,
        "train_jsonl": args.train_jsonl,
        "train_rows": len(ds),
        "max_seq_length": args.max_seq_length,
        "num_epochs": args.num_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "tokenization": {
            "encode_with_messages_format": True,
            "include_response": True,
            "response_only": False,
            "only_first_two": True,
            "prompt_only": False,
            "add_bos_token": False,
        },
        "dataloader": {
            "shuffle": (not args.no_shuffle_train),
        },
    }
    (output_dir / "training_config.json").write_text(json.dumps(cfg, indent=2))

    print("Done")


if __name__ == "__main__":
    main()
