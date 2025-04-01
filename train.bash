#!/bin/bash

# Parse command line arguments
MODEL_TYPE="r1"  # Default to r1
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

# Run training script
# Check GPU memory and choose appropriate config
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')

source .env
module load python && source .venv/bin/activate

# find hf token in env variables
HUGGINGFACE_TOKEN=$(printenv HUGGINGFACE_TOKEN)
echo "hf token: $HUGGINGFACE_TOKEN"

if [ "$MODEL_TYPE" = "qwen" ]; then
    if [ "$GPU_MEM" -ge 80000 ]; then
        # Use 80GB config for A100 or H100 with 80GB
        echo "Using 80GB config for Qwen"
        python distill_llama.py --model_config distill_deepscaler_1_5b_lk_smd_wtk64_fd64_w01 \
            --distill_config 80GB_distill_OpenR1_deepscaler_math \
            --finetune_config deepscaler_finetune_lora_qkvo_reasoning_dataset \
            --eval_config eval_alpaca_clean \
            --lk_zero_init \
            --verbose --seed 0 --replicate 0 \
            --huggingface_token "$HUGGINGFACE_TOKEN"
    else
        # Use 40GB config for smaller GPUs
        echo "Using 40GB config for Qwen"
        python distill_llama.py --model_config distill_deepscaler_1_5b_lk_smd_wtk64_fd64_w01 \
            --distill_config 40GB_distill_OpenR1_deepscaler_math \
            --finetune_config deepscaler_finetune_lora_qkvo_reasoning_dataset \
            --eval_config eval_alpaca_clean \
            --lk_zero_init \
            --verbose --seed 0 --replicate 0 \
            --huggingface_token "$HUGGINGFACE_TOKEN"
    fi
else
    if [ "$GPU_MEM" -ge 80000 ]; then
        # Use 80GB config for A100 or H100 with 80GB
        echo "Using 80GB config for R1"
        python distill_llama.py --model_config r1_distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
            --distill_config 80GB_distill_OpenR1_math \
            --finetune_config r1_finetune_lora_qkvo_reasoning_dataset \
            --eval_config eval_alpaca_clean \
            --lk_zero_init \
            --verbose --seed 0 --replicate 0 \
            --huggingface_token "$HUGGINGFACE_TOKEN"
    else
        # Use 40GB config for smaller GPUs
        echo "Using 40GB config for R1"
        python distill_llama.py --model_config r1_distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
            --distill_config 40GB_distill_OpenR1_math \
            --finetune_config r1_finetune_lora_qkvo_reasoning_dataset \
            --eval_config eval_alpaca_clean \
            --lk_zero_init \
            --verbose --seed 0 --replicate 0 \
            --huggingface_token "$HUGGINGFACE_TOKEN"
    fi
fi