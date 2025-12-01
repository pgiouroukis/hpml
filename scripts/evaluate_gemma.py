#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_gemma import run_test_evaluation, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Gemma model on the FinQA test set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path or identifier of the model checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional tokenizer path; defaults to model_path.")
    parser.add_argument("--test_file", type=str, default="FinQA/dataset/test.json")
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--input_mode", type=str, choices={"gold", "all"}, default="gold")
    parser.add_argument("--target_field", type=str, choices={"program", "exe_ans", "answer"}, default="program")
    parser.add_argument("--task_instruction", type=str, default="")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--generation_max_new_tokens", type=int, default=128)
    parser.add_argument("--generation_num_beams", type=int, default=1)
    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument("--generation_top_p", type=float, default=1.0)
    parser.add_argument("--generation_do_sample", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices={"eager", "sdpa", "flash_attention_2"},
        default="eager",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional name for the evaluation output folder (created under __output__).",
    )
    return parser.parse_args()


def build_output_dir(args: argparse.Namespace) -> Path:
    base = Path("__output__")
    base.mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        name = slugify(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"eval_{slugify(Path(args.model_path).name)}_{stamp}"
    output_dir = base / name
    suffix = 1
    while output_dir.exists():
        suffix += 1
        output_dir = base / f"{name}--{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args)
    tokenizer_name = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, attn_implementation=args.attn_implementation
    )
    metrics = run_test_evaluation(model=model, tokenizer=tokenizer, args=args, experiment_dir=output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
