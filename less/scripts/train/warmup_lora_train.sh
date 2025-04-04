#!/bin/bash

source less/scripts/train/base_training_args.sh

data_dir=$1
model_path=$2
percentage=$3
data_seed=$4
job_name=$5

output_dir=../out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

train_files=(
    "$data_dir/train/processed/cot/cot_data.jsonl"
    "$data_dir/train/processed/dolly/dolly_data.jsonl"
)

# use fsdp for large models
if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
    elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
fi

training_args="$base_training_args \
--data_dir $data_dir \
--model_name_or_path $model_path \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--train_files ${train_files[@]} \
--method $6 \
--per_device_train_batch_size $7 \
--subject $8 \
--n_val $9 \
--analysis_dataset ${10} \
--lora_target_modules ${11} \
--lora_r ${12} \
--lora_alpha ${13} \
--learning_rate ${14} \
--gradient_accumulation_steps ${15} \
--seed ${16} \
--subject ${17} 2>&1 | tee $output_dir/train.log" 


echo "Header: $header"
echo "Training Args: $training_args"

eval "$header" "$training_args"