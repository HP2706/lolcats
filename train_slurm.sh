#!/bin/bash
#SBATCH --job-name=lolcats_train
#SBATCH --output=lolcats_train_%j.out
#SBATCH --error=lolcats_train_%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# Parse command line arguments
MODEL_TYPE="qwen"  # Default to r1
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Update job name based on model type
#SBATCH --job-name=lolcats_${MODEL_TYPE}_train

bash env_setup.bash
bash train.bash --model_type ${MODEL_TYPE}