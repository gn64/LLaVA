#!/bin/bash

total_epochs=10
weights_save_limit=3
output_dir="./checkpoints/llava_test_output"
training_model="liuhaotian/llava-v1.5-7b"
training_data_path="train_dataset_path.json"
#venv_training="source ~/.venv/~~~/bin/activate"
venv_training=""

training_args="--lora_enable True --lora_r 128 --lora_alpha 256 --deepspeed ./scripts/zero3.json --version v1 --image_folder / --vision_tower openai/clip-vit-large-patch14-336 --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 --mm_use_im_start_end False --mm_use_im_patch_token False --image_aspect_ratio pad --group_by_modality_length True --bf16 True --per_device_train_batch_size 8 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --group_by_modality_length False --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --report_to wandb"

eval_model=llava-hf/llava-1.5-7b-hf
eval_data_path="evaluate_dataset_path.json"
eval_output_path="custom_metrics_script_output.json"
#venv_eval="source ~/.venv/~~~/bin/activate"
#venv_eval=""


for (( i=1; i<=total_epochs; i++ ))
do
  if [[ -n "$venv_training" ]]; then
    echo "Activating training environment..."
    eval "$venv_training"
  fi
  echo "Training Epoch $i/$total_epochs"
  deepspeed llava/train/train_mem.py --save_strategy "epoch" --num_train_epochs $i --model_name_or_path $training_model --data_path $training_data_path --validation_data_path $eval_data_path --output_dir $output_dir --save_total_limit $weights_save_limit $training_args
  if [ $? -ne 0 ]; then
    echo "Training failed at epoch $i."
    exit 1
  fi
  echo "Training Epoch $i/$total_epochs Done"
  echo "Evaluationg..."
  if [[ -n "$venv_eval" ]]; then
    echo "Activating training environment..."
    eval "$venv_eval"
  fi
  python llava/eval/custom_metrics/eval_custom_metrics.py --model_id $eval_model --checkpoint_folder $output_dir --data_path $eval_data_path --metrics_output_path $eval_output_path --epoch $i
  if [ $? -ne 0 ]; then
    echo "Evaluation failed at epoch $i."
    exit 1
  fi

done