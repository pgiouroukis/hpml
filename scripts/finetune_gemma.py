#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from sympy import simplify
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from tqdm import tqdm

try:
    from accelerate.optimizer import AcceleratedOptimizer
except ImportError:  # pragma: no cover
    AcceleratedOptimizer = None

if AcceleratedOptimizer is not None:

    def _safe_train(self):
        inner = getattr(self.optimizer, "train", None)
        if callable(inner):
            return inner()
        return None

    def _safe_eval(self):
        inner = getattr(self.optimizer, "eval", None)
        if callable(inner):
            return inner()
        return None

    AcceleratedOptimizer.train = _safe_train  # type: ignore[assignment]
    AcceleratedOptimizer.eval = _safe_eval  # type: ignore[assignment]


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"cannot interpret boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune Gemma-3-1B-IT on FinQA.")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--train_file", type=str, default="FinQA/dataset/train.json")
    parser.add_argument("--eval_file", type=str, default="FinQA/dataset/dev.json")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="",
        help="Optional name for the run directory inside __output__.",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        choices={"gold", "all", "noisy_gold"},
        default="gold",
        help="gold: only annotated evidence; all: full page; noisy_gold: evidence plus distractors.",
    )
    parser.add_argument(
        "--target_field",
        type=str,
        choices={"program", "exe_ans", "answer"},
        default="program",
    )
    parser.add_argument("--task_instruction", type=str, default="")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices={"eager", "sdpa", "flash_attention_2"},
        default="eager",
    )
    parser.add_argument(
        "--eval_strategy",
        "--evaluation_strategy",
        dest="eval_strategy",
        type=str,
        choices={"no", "steps", "epoch"},
        default="steps",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        choices={"steps", "epoch"},
        default="steps",
    )
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--bf16", type=str2bool, default=False)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=True)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--do_test", action="store_true", help="Run FinQA evaluation on the test split.")
    parser.add_argument("--test_file", type=str, default="FinQA/dataset/test.json")
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate for predictions.",
    )
    parser.add_argument("--generation_num_beams", type=int, default=1)
    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument("--generation_top_p", type=float, default=1.0)
    parser.add_argument(
        "--generation_do_sample",
        action="store_true",
        help="Use sampling instead of greedy/beam decoding when generating test predictions.",
    )
    parser.add_argument(
        "--noisy_text_distractors",
        type=int,
        default=3,
        help="Number of random non-evidence sentences to add in noisy_gold mode (per pre/post section).",
    )
    parser.add_argument(
        "--noisy_table_distractors",
        type=int,
        default=1,
        help="Number of random non-evidence table rows to add in noisy_gold mode.",
    )
    parser.add_argument(
        "--noisy_context_seed",
        type=int,
        default=0,
        help="Base seed to make noisy_gold sampling deterministic per example.",
    )
    return parser.parse_args()


def table_to_lines(table: Sequence[Sequence[str]] | None) -> List[str]:
    if not table:
        return []
    lines: List[str] = []
    for row in table:
        cleaned = [cell.strip() for cell in row if cell and cell.strip()]
        if cleaned:
            lines.append(" | ".join(cleaned))
    return lines


def _sample_with_noise(
    lines: Sequence[str],
    gold_indices: Sequence[int],
    noise_k: int,
    rng: random.Random,
) -> List[str]:
    if not lines:
        return []
    gold_set = set(gold_indices)
    pool = [idx for idx in range(len(lines)) if idx not in gold_set]
    noise = rng.sample(pool, min(noise_k, len(pool))) if noise_k > 0 else []
    keep = sorted(gold_set.union(noise))
    return [lines[idx] for idx in keep]


def build_context(
    entry: Dict,
    mode: str,
    noisy_text_distractors: int = 0,
    noisy_table_distractors: int = 0,
    noisy_seed: int = 0,
    example_index: int = 0,
) -> str:
    qa = entry["qa"]
    if mode == "gold":
        gold = qa.get("gold_inds") or {}
        ordered = [gold[key].strip() for key in sorted(gold)]
        if ordered:
            return "\n".join(ordered)

    pre = entry.get("pre_text") or []
    post = entry.get("post_text") or []
    table_lines = table_to_lines(entry.get("table"))

    # Default: full context
    if mode == "all":
        parts: List[str] = []
        if pre:
            parts.append("PRE TEXT:\n" + "\n".join(pre))
        if table_lines:
            parts.append("TABLE:\n" + "\n".join(table_lines))
        if post:
            parts.append("POST TEXT:\n" + "\n".join(post))
        return "\n\n".join(parts) if parts else "No additional context provided."

    # noisy_gold
    rng = random.Random(noisy_seed + example_index)
    gold_text_rows: List[int] = qa.get("ann_text_rows") or []
    gold_table_rows: List[int] = qa.get("ann_table_rows") or []
    pre_len = len(pre)
    pre_gold = [idx for idx in gold_text_rows if idx < pre_len]
    post_gold = [idx - pre_len for idx in gold_text_rows if idx >= pre_len]

    pre_lines = _sample_with_noise(pre, pre_gold, noisy_text_distractors, rng)
    post_lines = _sample_with_noise(post, post_gold, noisy_text_distractors, rng)
    table_selected = _sample_with_noise(table_lines, gold_table_rows, noisy_table_distractors, rng)

    parts: List[str] = []
    if pre_lines:
        parts.append("PRE TEXT:\n" + "\n".join(pre_lines))
    if table_selected:
        parts.append("TABLE:\n" + "\n".join(table_selected))
    if post_lines:
        parts.append("POST TEXT:\n" + "\n".join(post_lines))
    return "\n\n".join(parts) if parts else "No additional context provided."


def build_prompt(question: str, context: str, hint: str) -> str:
    return (
        f"{hint.strip()}\n\nQuestion:\n{question.strip()}\n\nContext:\n{context.strip()}\n\nAnswer:"
    )


def extract_target(qa: Dict, field: str) -> str:
    target = qa.get(field, "")
    if isinstance(target, list):
        target = " ".join(target)
    if not target:
        fallback = qa.get("exe_ans") or qa.get("answer") or "unknown"
        target = fallback
    return str(target).strip()


def encode_example(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    target: str,
    max_length: int,
) -> Dict[str, List[int]]:
    target_text = (target or "unknown").strip()
    eos = tokenizer.eos_token or ""
    target_with_eos = f"{target_text}{eos}"
    target_ids = tokenizer(
        target_with_eos,
        add_special_tokens=False,
    ).input_ids
    if len(target_ids) >= max_length:
        target_ids = target_ids[-max_length:]
        prompt_ids: List[int] = []
    else:
        remaining = max_length - len(target_ids)
        prompt_ids = (
            tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=remaining,
            ).input_ids
            if remaining > 0
            else []
        )
    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class FinQADataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        input_mode: str,
        target_field: str,
        task_instruction: str,
        max_samples: int = 0,
        noisy_text_distractors: int = 0,
        noisy_table_distractors: int = 0,
        noisy_context_seed: int = 0,
    ):
        raw_data = json.loads(Path(path).read_text())
        if max_samples:
            raw_data = raw_data[:max_samples]
        hint = resolve_task_hint(target_field, task_instruction)
        self.features = []
        for idx, entry in enumerate(raw_data):
            context = build_context(
                entry,
                input_mode,
                noisy_text_distractors=noisy_text_distractors,
                noisy_table_distractors=noisy_table_distractors,
                noisy_seed=noisy_context_seed,
                example_index=idx,
            )
            self.features.append(
                encode_example(
                    tokenizer=tokenizer,
                    prompt=build_prompt(
                        entry["qa"]["question"],
                        context,
                        hint,
                    ),
                    target=extract_target(entry["qa"], target_field),
                    max_length=max_length,
                )
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.features[idx]


@dataclass
class LMDataCollator:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        for item in features:
            diff = max_length - len(item["input_ids"])
            batch_input_ids.append(item["input_ids"] + [pad_id] * diff)
            batch_attention.append(item["attention_mask"] + [0] * diff)
            batch_labels.append(item["labels"] + [self.label_pad_token_id] * diff)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


ALL_OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
]


def program_tokenization(original_program: str) -> List[str]:
    if not isinstance(original_program, str):
        original_program = str(original_program)
    original_program = original_program.strip()
    if not original_program:
        return ["EOF"]
    program: List[str] = []
    for tok in re.split(r",\s*", original_program):
        if not tok:
            continue
        cur_tok = ""
        for char in tok:
            if char == ")":
                if cur_tok:
                    program.append(cur_tok)
                    cur_tok = ""
            cur_tok += char
            if char in {"(", ")"}:
                program.append(cur_tok)
                cur_tok = ""
        if cur_tok:
            program.append(cur_tok)
    program.append("EOF")
    return program


def str_to_num(text: str):
    text = text.replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text) / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num


def process_row(row_in: Sequence[str]):
    row_out: List[float] = []
    for num in row_in:
        num = num.replace("$", "").strip()
        num = num.split("(")[0].strip()
        num_value = str_to_num(num)
        if num_value == "n/a":
            return "n/a"
        row_out.append(num_value)
    return row_out


def eval_program(program: Sequence[str], table: Sequence[Sequence[str]] | None):
    invalid_flag = 0
    this_res = "n/a"
    try:
        program = list(program[:-1])  # remove EOF
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        program_str = "|".join(program)
        steps = program_str.split(")")[:-1]
        res_dict = {}

        for ind, step in enumerate(steps):
            step = step.strip()
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if op in {"add", "subtract", "multiply", "divide", "exp", "greater"}:
                if "#" in arg1:
                    arg1_val = res_dict.get(int(arg1.replace("#", "")), "n/a")
                else:
                    arg1_val = str_to_num(arg1)
                if arg1_val == "n/a":
                    invalid_flag = 1
                    break

                if "#" in arg2:
                    arg2_val = res_dict.get(int(arg2.replace("#", "")), "n/a")
                else:
                    arg2_val = str_to_num(arg2)
                if arg2_val == "n/a":
                    invalid_flag = 1
                    break

                if op == "add":
                    this_res = arg1_val + arg2_val
                elif op == "subtract":
                    this_res = arg1_val - arg2_val
                elif op == "multiply":
                    this_res = arg1_val * arg2_val
                elif op == "divide":
                    this_res = arg1_val / arg2_val
                elif op == "exp":
                    this_res = arg1_val**arg2_val
                elif op == "greater":
                    this_res = "yes" if arg1_val > arg2_val else "no"
                res_dict[ind] = this_res
            elif "table" in op and table is not None:
                table_dict = {row[0]: row[1:] for row in table}
                if "#" in arg1:
                    arg1_val = res_dict.get(int(arg1.replace("#", "")), "n/a")
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    num_row = process_row(table_dict[arg1])
                    arg1_val = num_row

                if arg1_val == "n/a":
                    invalid_flag = 1
                    break
                if op == "table_max":
                    this_res = max(arg1_val)
                elif op == "table_min":
                    this_res = min(arg1_val)
                elif op == "table_sum":
                    this_res = sum(arg1_val)
                elif op == "table_average":
                    this_res = sum(arg1_val) / len(arg1_val)
                res_dict[ind] = this_res
            else:
                invalid_flag = 1
                break

        if this_res not in {"yes", "no", "n/a"}:
            this_res = round(this_res, 5)
    except Exception:
        invalid_flag = 1
    return invalid_flag, this_res


def equal_program(program1: Sequence[str], program2: Sequence[str]) -> bool:
    sym_map: Dict[str, str] = {}
    program1 = program1[:-1]
    program1_str = "|".join(program1)
    steps = program1_str.split(")")[:-1]
    step_dict_1: Dict[int, str] = {}
    sym_ind = 0

    for ind, step in enumerate(steps):
        step = step.strip()
        assert len(step.split("(")) <= 2
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()
        step_dict_1[ind] = step
        if "table" in op:
            if step not in sym_map:
                sym_map[step] = f"a{sym_ind}"
                sym_ind += 1
        else:
            if "#" not in arg1 and arg1 not in sym_map:
                sym_map[arg1] = f"a{sym_ind}"
                sym_ind += 1
            if "#" not in arg2 and arg2 not in sym_map:
                sym_map[arg2] = f"a{sym_ind}"
                sym_ind += 1

    step_dict_2: Dict[int, str] = {}
    try:
        program2 = program2[:-1]
        for ind, token in enumerate(program2):
            if ind % 4 == 0 and token.strip("(") not in ALL_OPS:
                return False
            if (ind + 1) % 4 == 0 and token != ")":
                return False
        program2_str = "|".join(program2)
        steps = program2_str.split(")")[:-1]
        for ind, step in enumerate(steps):
            step = step.strip()
            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            step_dict_2[ind] = step
            if "table" in op:
                if step not in sym_map:
                    return False
            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                elif int(arg1.strip("#")) >= ind:
                    return False
                if "#" not in arg2:
                    if arg2 not in sym_map:
                        return False
                elif int(arg2.strip("#")) >= ind:
                    return False
    except Exception:
        return False

    def symbol_recur(step: str, step_dict: Dict[int, str]) -> str:
        step = step.strip()
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()
        if "table" in op:
            return sym_map[step]
        if "#" in arg1:
            arg1_part = symbol_recur(step_dict[int(arg1.replace("#", ""))], step_dict)
        else:
            arg1_part = sym_map[arg1]
        if "#" in arg2:
            arg2_part = symbol_recur(step_dict[int(arg2.replace("#", ""))], step_dict)
        else:
            arg2_part = sym_map[arg2]
        if op == "add":
            return f"( {arg1_part} + {arg2_part} )"
        if op == "subtract":
            return f"( {arg1_part} - {arg2_part} )"
        if op == "multiply":
            return f"( {arg1_part} * {arg2_part} )"
        if op == "divide":
            return f"( {arg1_part} / {arg2_part} )"
        if op == "exp":
            return f"( {arg1_part} ** {arg2_part} )"
        if op == "greater":
            return f"( {arg1_part} > {arg2_part} )"
        return ""

    steps_prog1 = program1_str.split(")")[:-1]
    sym_prog1 = symbol_recur(steps_prog1[-1], step_dict_1)
    sym_prog1 = simplify(sym_prog1, evaluate=False)
    try:
        steps_prog2 = program2_str.split(")")[:-1]
        sym_prog2 = symbol_recur(steps_prog2[-1], step_dict_2)
        sym_prog2 = simplify(sym_prog2, evaluate=False)
    except Exception:
        return False
    return sym_prog1 == sym_prog2


def clean_prediction_text(text: str, eos_token: str | None = None) -> str:
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.strip()
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:", 1)[-1].strip()
    if "\n" in cleaned:
        cleaned = cleaned.split("\n", 1)[0].strip()
    if eos_token and eos_token in cleaned:
        cleaned = cleaned.split(eos_token, 1)[0].strip()
    return cleaned


def evaluate_program_prediction(
    prediction: str,
    gold_program: str,
    gold_exe_ans,
    table: Sequence[Sequence[str]] | None,
) -> Dict:
    pred_tokens = program_tokenization(prediction)
    gold_tokens = program_tokenization(gold_program)
    invalid_flag, exe_res = eval_program(pred_tokens, table or [])
    exec_error = None
    execution_accuracy = invalid_flag == 0 and exe_res == gold_exe_ans
    try:
        program_accuracy = equal_program(gold_tokens, pred_tokens)
    except Exception as exc:  # pragma: no cover - defensive
        program_accuracy = False
        exec_error = f"program_equivalence_failed: {exc}"
    if invalid_flag:
        exec_error = exec_error or "invalid program or execution failure"
    if program_accuracy and invalid_flag == 0 and exe_res != gold_exe_ans:
        exec_error = exec_error or "equivalent program but execution mismatch"
    return {
        "predicted_exe_ans": None if invalid_flag else exe_res,
        "execution_accuracy": execution_accuracy,
        "program_accuracy": program_accuracy,
        "execution_error": exec_error,
    }


def load_finqa_split(path: str, max_samples: int = 0) -> List[Dict]:
    raw_data = json.loads(Path(path).read_text())
    if max_samples:
        raw_data = raw_data[:max_samples]
    return raw_data


def resolve_task_hint(target_field: str, task_instruction: str) -> str:
    hint = task_instruction.strip()
    if not hint:
        if target_field == "program":
            hint = "Generate the FinQA reasoning program (DSL) that answers the question."
        else:
            hint = "Return the final numeric answer."
    return hint


def build_generation_entries(
    data: List[Dict],
    input_mode: str,
    target_field: str,
    task_instruction: str,
    noisy_text_distractors: int = 0,
    noisy_table_distractors: int = 0,
    noisy_context_seed: int = 0,
) -> List[Dict]:
    hint = resolve_task_hint(target_field, task_instruction)
    entries: List[Dict] = []
    for idx, entry in enumerate(data):
        qa = entry["qa"]
        prompt = build_prompt(
            qa["question"],
            build_context(
                entry,
                input_mode,
                noisy_text_distractors=noisy_text_distractors,
                noisy_table_distractors=noisy_table_distractors,
                noisy_seed=noisy_context_seed,
                example_index=idx,
            ),
            hint,
        )
        entries.append(
            {
                "id": entry["id"],
                "qa": qa,
                "table": entry.get("table"),
                "prompt": prompt,
            }
        )
    return entries


def generate_predictions_for_entries(
    model,
    tokenizer: PreTrainedTokenizerBase,
    entries: List[Dict],
    max_seq_length: int,
    generation_kwargs: Dict,
) -> tuple[List[Dict], Dict[str, float]]:
    device = model.device if hasattr(model, "device") else torch.device("cpu")
    if getattr(model.config, "use_cache", None) is False:
        model.config.use_cache = True
    model.eval()
    predictions: List[Dict] = []
    latencies_ms: List[float] = []
    overall_start = time.perf_counter()
    for item in tqdm(entries, desc="Generating on test split"):
        encoded = tokenizer(
            item["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                **generation_kwargs,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        generated_tokens = output_ids[0][encoded["input_ids"].shape[1] :]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        prediction_text = clean_prediction_text(decoded, tokenizer.eos_token)
        predictions.append(
            {
                **item,
                "prediction": prediction_text,
                "latency_ms": latency_ms,
            }
        )
        latencies_ms.append(latency_ms)
    total_seconds = time.perf_counter() - overall_start
    return predictions, {"total_seconds": total_seconds, "latencies_ms": latencies_ms}


def score_predictions(
    predictions: List[Dict],
    target_field: str,
) -> tuple[List[Dict], Dict[str, float]]:
    scored: List[Dict] = []
    exec_correct = 0
    prog_correct = 0
    answer_correct = 0
    total = len(predictions)
    for item in predictions:
        qa = item["qa"]
        base_record = {
            "id": item["id"],
            "question": qa.get("question", ""),
            "target_field": target_field,
            "prompt": item["prompt"],
            "prediction": item["prediction"],
        }
        if target_field == "program":
            gold_program = qa.get("program", "")
            gold_exe_ans = qa.get("exe_ans")
            metrics = evaluate_program_prediction(
                prediction=item["prediction"],
                gold_program=gold_program,
                gold_exe_ans=gold_exe_ans,
                table=item.get("table") or [],
            )
            base_record.update(
                {
                    "gold_program": gold_program,
                    "gold_exe_ans": gold_exe_ans,
                    "predicted_exe_ans": metrics["predicted_exe_ans"],
                    "execution_accuracy": metrics["execution_accuracy"],
                    "program_accuracy": metrics["program_accuracy"],
                    "execution_error": metrics["execution_error"],
                }
            )
            exec_correct += int(bool(metrics["execution_accuracy"]))
            prog_correct += int(bool(metrics["program_accuracy"]))
        else:
            gold_target = extract_target(qa, target_field)
            answer_match = item["prediction"].strip() == gold_target
            base_record.update(
                {
                    "gold_target": gold_target,
                    "answer_accuracy": answer_match,
                }
            )
            answer_correct += int(bool(answer_match))
        scored.append(base_record)

    aggregate: Dict[str, float] = {
        "total": total,
        "target_field": target_field,
    }
    if target_field == "program":
        aggregate["execution_accuracy"] = exec_correct / total if total else 0.0
        aggregate["program_accuracy"] = prog_correct / total if total else 0.0
    else:
        aggregate["answer_accuracy"] = answer_correct / total if total else 0.0
    return scored, aggregate


def run_test_evaluation(model, tokenizer, args: argparse.Namespace, experiment_dir: Path) -> Dict:
    test_data = load_finqa_split(args.test_file, args.max_test_samples)
    generation_entries = build_generation_entries(
        data=test_data,
        input_mode=args.input_mode,
        target_field=args.target_field,
        task_instruction=args.task_instruction,
        noisy_text_distractors=args.noisy_text_distractors,
        noisy_table_distractors=args.noisy_table_distractors,
        noisy_context_seed=args.noisy_context_seed,
    )
    generation_kwargs = {
        "max_new_tokens": args.generation_max_new_tokens,
        "num_beams": args.generation_num_beams,
        "temperature": args.generation_temperature,
        "top_p": args.generation_top_p,
        "do_sample": args.generation_do_sample,
    }
    predictions, timing_info = generate_predictions_for_entries(
        model=model,
        tokenizer=tokenizer,
        entries=generation_entries,
        max_seq_length=args.max_seq_length,
        generation_kwargs=generation_kwargs,
    )
    scored, aggregate = score_predictions(predictions, args.target_field)

    lat_path = experiment_dir / "test_inference_latencies.jsonl"
    with lat_path.open("w", encoding="utf-8") as handle:
        for pred in predictions:
            handle.write(json.dumps({"id": pred["id"], "latency_ms": pred.get("latency_ms")}) + "\n")

    latencies = timing_info.get("latencies_ms", [])
    if latencies:
        lat_sorted = sorted(latencies)
        n = len(lat_sorted)

        def percentile(p: float) -> float:
            if not lat_sorted:
                return 0.0
            k = (n - 1) * (p / 100.0)
            f = int(k)
            c = min(f + 1, n - 1)
            if f == c:
                return lat_sorted[f]
            return lat_sorted[f] + (lat_sorted[c] - lat_sorted[f]) * (k - f)

        aggregate.update(
            {
                "inference_total_seconds": timing_info.get("total_seconds", 0.0),
                "inference_latency_ms_avg": sum(latencies) / n,
                "inference_latency_ms_p50": percentile(50),
                "inference_latency_ms_p90": percentile(90),
                "inference_latency_ms_p95": percentile(95),
                "inference_latency_ms_p99": percentile(99),
                "latency_file": str(lat_path),
            }
        )

    predictions_path = experiment_dir / "test_predictions.jsonl"
    metrics_path = experiment_dir / "test_metrics.json"

    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row) + "\n")

    aggregate.update(
        {
            "input_mode": args.input_mode,
            "output_jsonl": str(predictions_path),
            "generation_max_new_tokens": args.generation_max_new_tokens,
            "generation_num_beams": args.generation_num_beams,
            "generation_temperature": args.generation_temperature,
            "generation_top_p": args.generation_top_p,
            "generation_do_sample": args.generation_do_sample,
            "max_seq_length": args.max_seq_length,
            "test_file": args.test_file,
        }
    )
    metrics_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def slugify(value: str) -> str:
    cleaned = value.replace(os.sep, "-").replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    result = "".join(ch if ch in allowed else "-" for ch in cleaned)
    result = result.strip("-_.")
    return result or "run"


def build_experiment_dir(args: argparse.Namespace) -> Path:
    base = Path("__output__")
    base.mkdir(parents=True, exist_ok=True)
    custom = (getattr(args, "output_folder", "") or "").strip()
    if custom:
        name = slugify(custom)
        experiment_dir = base / name
        suffix = 1
        while experiment_dir.exists():
            suffix += 1
            experiment_dir = base / f"{name}--{suffix}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
    label = slugify(Path(args.model_name).name)
    precision_tokens = []
    if args.bf16:
        precision_tokens.append("bf16")
    if args.fp16 and not args.bf16:
        precision_tokens.append("fp16")
    if not precision_tokens:
        precision_tokens.append("fp32")
    precision_str = "+".join(precision_tokens)
    parts = [
        label,
        f"t{slugify(args.target_field)}",
        f"i{slugify(args.input_mode)}",
        f"seq{args.max_seq_length}",
        f"bs{args.per_device_train_batch_size}",
        f"ep{str(args.num_train_epochs).replace('.', '_')}",
        f"prec{precision_str}",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    ]
    name = "_".join(parts)
    experiment_dir = base / name
    suffix = 1
    while experiment_dir.exists():
        suffix += 1
        experiment_dir = base / f"{name}--{suffix}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


class FileLoggerCallback(TrainerCallback):
    def __init__(
        self,
        metrics_file: Path,
        train_loss_file: Path,
        eval_loss_file: Path,
    ):
        self.metrics_file = metrics_file
        self.train_loss_file = train_loss_file
        self.eval_loss_file = eval_loss_file
        for target in (self.metrics_file, self.train_loss_file, self.eval_loss_file):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return int(value)
            except (TypeError, ValueError):
                return str(value)

    def _write_metrics(self, record: Dict):
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _write_plain(self, path: Path, step: int, epoch: float, value_name: str, value):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{step}\t{epoch}\t{value_name}\t{value}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {
            "type": "train",
            "step": state.global_step,
            "epoch": state.epoch,
        }
        record.update({k: self._to_serializable(v) for k, v in logs.items()})
        self._write_metrics(record)
        if "loss" in logs:
            self._write_plain(
                self.train_loss_file,
                state.global_step,
                state.epoch if state.epoch is not None else 0,
                "loss",
                logs["loss"],
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        record = {
            "type": "eval",
            "step": state.global_step,
            "epoch": state.epoch,
        }
        record.update({k: self._to_serializable(v) for k, v in metrics.items()})
        self._write_metrics(record)
        if "eval_loss" in metrics:
            self._write_plain(
                self.eval_loss_file,
                state.global_step,
                state.epoch if state.epoch is not None else 0,
                "eval_loss",
                metrics["eval_loss"],
            )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    precision = "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32")
    experiment_dir = build_experiment_dir(args)
    with (experiment_dir / "logs.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} params_precision={precision}\n")
    final_model_dir = experiment_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["resolved_output_dir"] = str(experiment_dir)
    config_path = experiment_dir / "args.json"
    config_path.write_text(json.dumps(args_dict, indent=2, sort_keys=True))
    metrics_file = experiment_dir / "metrics.jsonl"
    train_loss_file = experiment_dir / "train_loss.log"
    eval_loss_file = experiment_dir / "eval_loss.log"

    print(f"Experiment directory: {experiment_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = FinQADataset(
        path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        input_mode=args.input_mode,
        target_field=args.target_field,
        task_instruction=args.task_instruction,
        max_samples=args.max_train_samples,
        noisy_text_distractors=args.noisy_text_distractors,
        noisy_table_distractors=args.noisy_table_distractors,
        noisy_context_seed=args.noisy_context_seed,
    )

    eval_dataset = (
        FinQADataset(
            path=args.eval_file,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            input_mode=args.input_mode,
            target_field=args.target_field,
            task_instruction=args.task_instruction,
            max_samples=args.max_eval_samples,
            noisy_text_distractors=args.noisy_text_distractors,
            noisy_table_distractors=args.noisy_table_distractors,
            noisy_context_seed=args.noisy_context_seed,
        )
        if args.do_eval
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, attn_implementation=args.attn_implementation
    )
    model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(experiment_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy if args.do_eval else "no",
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="none",
        load_best_model_at_end=args.load_best_model_at_end and args.do_eval,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=LMDataCollator(tokenizer),
        callbacks=[
            FileLoggerCallback(
                metrics_file=metrics_file,
                train_loss_file=train_loss_file,
                eval_loss_file=eval_loss_file,
            )
        ],
    )

    timings: Dict[str, float] = {}
    train_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    timings["train_duration_seconds"] = time.perf_counter() - train_start
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(final_model_dir)

    if args.do_eval and eval_dataset is not None:
        eval_start = time.perf_counter()
        metrics = trainer.evaluate()
        timings["eval_duration_seconds"] = time.perf_counter() - eval_start
        metrics["eval_duration_seconds"] = timings["eval_duration_seconds"]
        print(metrics)

    if args.do_test:
        test_metrics = run_test_evaluation(
            model=model,
            tokenizer=tokenizer,
            args=args,
            experiment_dir=experiment_dir,
        )
        timings["test_inference_seconds"] = test_metrics.get("inference_total_seconds", 0.0)
        print("Test metrics:", json.dumps(test_metrics, indent=2))

    # persist timing summary
    timing_path = experiment_dir / "timing.json"
    timing_path.write_text(json.dumps(timings, indent=2, sort_keys=True))

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
