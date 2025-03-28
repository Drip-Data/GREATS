#!/bin/bash
#SBATCH --job-name=online-grad-select
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=11:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

#SBATCH --output=/scratch/gpfs/tw8948/slurm-%j.out
#SBATCH --error=/scratch/gpfs/tw8948/slurm-%j.out

DATA_DIR=./data/data
MODEL_PATH=../models/Llama-3.2-1B
DATA_SEED=3
LORA_R=128
LORA_ALPHA=512



method=$1
batch_size=$2
PERCENTAGE=$3 # percentage of the full data to train, you can specify the training file you want to use in the script
NVAL=$4
task=$5
lr=$6
seed=${7:-"42"}
gradient_accumulation_steps=${8:-"1"}
subject=${9:-"world_religions"}

JOB_NAME=llama3-1b-p${PERCENTAGE}-lora-r${LORA_R}-lora_alpha${LORA_ALPHA}-seed${DATA_SEED}

# Set combined_modules based on the task
if [ "$task" = "mmlu" ]; then
    combined_modules="q_proj k_proj v_proj o_proj"  
else
    combined_modules="q_proj k_proj" 
fi


./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$method" "$batch_size" "$subject" "$NVAL" "$task" "$combined_modules" "$LORA_R" "$LORA_ALPHA" "$lr" "$gradient_accumulation_steps" "$seed" "$subject"

