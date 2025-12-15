# FinQA Finetuning Utilities

This repository contains scripts to finetune instruction-tuned causal LMs (e.g., Gemma and Qwen) on the FinQA dataset, generate FinQA DSL programs or numeric answers, and evaluate execution accuracy. It also includes utilities for noisy-evidence training and runtime tracking.

## Key scripts
- `scripts/finetune_gemma.py`: Main training/eval driver.
  - Modes:
    - `--target_field program|numerical|answer` (DSL program vs executed numeric answer vs raw answer string).
    - `--input_mode gold|all|noisy_gold`:
      - `gold`: only annotated evidence (`gold_inds`); oracle context.
      - `all`: full pre_text/table/post_text.
      - `noisy_gold`: annotated evidence plus random distractors (see noise knobs below).
  - Noise knobs (for `noisy_gold`):
    - `--noisy_text_distractors`: number of non-evidence pre/post sentences to add.
    - `--noisy_table_distractors`: number of non-evidence table rows to add.
    - `--noisy_context_seed`: base seed for deterministic sampling per example.
  - Model knobs:
    - `--model_name`: Hugging Face id or local path (e.g., `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `Qwen/Qwen2.5-Coder-3B-Instruct`).
    - `--peft none|lora|qlora`: full finetune vs LoRA vs 4-bit QLoRA (recommended for 4B).
    - LoRA knobs: `--lora_r`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`.
    - QLoRA knobs: `--bnb_4bit_quant_type`, `--bnb_4bit_compute_dtype`, `--bnb_4bit_use_double_quant`.
    - Optional `--optim` override (e.g., `paged_adamw_8bit`).
  - Generation knobs: `--generation_max_new_tokens`, `--generation_num_beams`, `--generation_temperature`, `--generation_top_p`, `--generation_do_sample`.
  - Training knobs: standard HF Trainer arguments (epochs, LR, batch sizes, accumulation, scheduler, precision, etc.).
  - Outputs per run (under `__output__/.../`):
    - `args.json`: resolved arguments.
    - `metrics.jsonl`, `train_loss.log`, `eval_loss.log`: training/eval traces.
    - `final_model/`: saved model and tokenizer.
    - `test_predictions.jsonl`: per-example predictions (includes DSL/answer, prompt, and per-sample inference latency in ms).
    - `test_metrics.json`: aggregate metrics (execution/program/answer accuracy plus inference timing stats).
    - `test_inference_latencies.jsonl`: per-example generation latency in milliseconds.
    - `timing.json`: wall-clock durations for train/eval/test inference.
- `scripts/make_noisy_gold_dataset.py`: Materializes prompts with noisy-gold context for offline inspection/training. Writes JSONL with `id`, `question`, `context`, `prompt`, and `target`.

## Runtime tracking
- Training time: recorded as `train_duration_seconds` in `timing.json`.
- Eval time: `eval_duration_seconds` in `timing.json` and echoed into eval metrics.
- Test inference:
  - Per-sample latency (ms) is stored in `test_predictions.jsonl` and in `test_inference_latencies.jsonl`.
  - Aggregates in `test_metrics.json`: `inference_total_seconds`, `inference_latency_ms_{avg,p50,p90,p95,p99}`, and pointer to `latency_file`.
- GPU memory: `cuda_max_memory_bytes` is recorded in `timing.json` when CUDA is available.
- Optional W&B logging: set `--wandb_project` and `--wandb_mode` to log all metrics (train/eval/test, timings, latency histogram) to Weights & Biases.

## Ready-made run scripts (`__scripts__/`)
- `gemma-input_gold-output_program/script.sh`: Oracle-evidence DSL training/eval (`input_mode gold`, `target_field program`).
- `gemma-input_all-output_program/script.sh`: Full-context DSL training/eval (`input_mode all`, `target_field program`).
- `gemma-input_noisy_gold-output_program/script.sh`: DSL training/eval with noisy-gold context (`input_mode noisy_gold`, distractor defaults set).
- `gemma-input_gold-output_numerical/script.sh`: Oracle-evidence numeric-answer training/eval (`input_mode gold`, `target_field numerical`).
- `gemma-input_all-output_numerical/script.sh`: Full-context numeric-answer training/eval (`input_mode all`, `target_field numerical`).
- `gemma3-4b-qlora-input_gold-output_program/script.sh`: Gemma-3-4B-IT QLoRA DSL training/eval (`input_mode gold`, `target_field program`).
- `gemma3-4b-qlora-input_all-output_program/script.sh`: Gemma-3-4B-IT QLoRA DSL training/eval (`input_mode all`, `target_field program`).
- `gemma3-4b-qlora-input_noisy_gold-output_program/script.sh`: Gemma-3-4B-IT QLoRA DSL training/eval (`input_mode noisy_gold`, distractor defaults set).
- `gemma3-4b-qlora-input_gold-output_numerical/script.sh`: Gemma-3-4B-IT QLoRA numeric-answer training/eval (`input_mode gold`, `target_field numerical`).
- `gemma3-4b-qlora-input_all-output_numerical/script.sh`: Gemma-3-4B-IT QLoRA numeric-answer training/eval (`input_mode all`, `target_field numerical`).
- `qwen2-5-coder-3b-instruct-full-input_gold-output_program/script.sh`: Qwen2.5-Coder-3B full finetune DSL training/eval (`input_mode gold`, `target_field program`).
- `qwen2-5-coder-3b-instruct-full-input_all-output_program/script.sh`: Qwen2.5-Coder-3B full finetune DSL training/eval (`input_mode all`, `target_field program`).
- `qwen2-5-coder-3b-instruct-full-input_noisy_gold-output_program/script.sh`: Qwen2.5-Coder-3B full finetune DSL training/eval (`input_mode noisy_gold`, distractor defaults set).
- `qwen2-5-coder-3b-instruct-full-input_gold-output_numerical/script.sh`: Qwen2.5-Coder-3B full finetune numeric-answer training/eval (`input_mode gold`, `target_field numerical`).
- `qwen2-5-coder-3b-instruct-full-input_all-output_numerical/script.sh`: Qwen2.5-Coder-3B full finetune numeric-answer training/eval (`input_mode all`, `target_field numerical`).

Run any script with `bash __scripts__/<run>/script.sh`. Outputs land under `__output__/<matching-name>*/`.

## Typical adjustments
- Shorten/expand context: switch `--input_mode` or adjust distractor counts.
- Precision: set `--bf16 true` or `--fp16 true` if your hardware supports it.
- Large models: use `--peft qlora` (and optionally `--optim paged_adamw_8bit`) to make Gemma-3-4B fit on a single GPU.
- Training budget: bump `--num_train_epochs`, `--per_device_train_batch_size`, or adjust `--learning_rate` as needed.
- Decoding: tweak beam/sampling settings via generation flags.

## Dataset note
FinQA splits live under `FinQA/dataset/` (`train.json`, `dev.json`, `test.json`). DSL definitions and evaluation follow the FinQA paper; execution accuracy is computed by running the predicted DSL against the normalized table/text fields.
