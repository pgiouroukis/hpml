#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_gemma import run_test_evaluation, slugify


def resolve_bnb_compute_dtype(args: argparse.Namespace) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    requested = getattr(args, "bnb_4bit_compute_dtype", "auto")
    if requested in mapping:
        return mapping[requested]
    if getattr(args, "bf16", False):
        return torch.bfloat16
    if getattr(args, "fp16", False):
        return torch.float16
    return torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Gemma model on the FinQA test set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path or identifier of the model checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional tokenizer path; defaults to model_path.")
    parser.add_argument("--test_file", type=str, default="FinQA/dataset/test.json")
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--input_mode", type=str, choices={"gold", "all"}, default="gold")
    parser.add_argument("--target_field", type=str, choices={"program", "numerical", "answer"}, default="program")
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
    parser.add_argument("--bf16", action="store_true", help="Load weights in bfloat16 when possible.")
    parser.add_argument("--fp16", action="store_true", help="Load weights in float16 when possible.")
    parser.add_argument(
        "--device_map",
        type=str,
        default="",
        help="Optional Hugging Face device_map for from_pretrained (e.g., 'auto').",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the base model in 4-bit (useful for large checkpoints).",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        choices={"nf4", "fp4"},
        default="nf4",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        choices={"auto", "bf16", "fp16", "fp32"},
        default="auto",
        help="Compute dtype for 4-bit quantization (auto uses bf16/fp16 flags, else fp16).",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action=argparse.BooleanOptionalAction,
        default=True,
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

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    device_map = (args.device_map or "").strip() or None
    quantization_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=resolve_bnb_compute_dtype(args),
        )
        if device_map is None:
            device_map = "auto"

    model_kwargs = {"attn_implementation": args.attn_implementation}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    adapter_config_path = Path(args.model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"PEFT adapter detected at {adapter_config_path}, but `peft` is not installed."
            ) from exc

        adapter_config = json.loads(adapter_config_path.read_text())
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(
                f"Missing base_model_name_or_path in {adapter_config_path}; cannot load base model."
            )
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    metrics = run_test_evaluation(model=model, tokenizer=tokenizer, args=args, experiment_dir=output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
