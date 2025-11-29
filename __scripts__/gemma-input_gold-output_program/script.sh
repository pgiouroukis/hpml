python scripts/finetune_gemma.py \
  --output_folder gemma-input_gold-output_program \
  --input_mode gold \
  --target_field program \
  --do_eval \
  --eval_steps 200 \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 4
