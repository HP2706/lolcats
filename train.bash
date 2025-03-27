# Run training script
# Check GPU memory and choose appropriate config
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')

module load python && source .venv/bin/activate
if [ "$GPU_MEM" -ge 80000 ]; then
    # Use 80GB config for A100 or H100 with 80GB
    echo "Using 80GB config"
    python distill_llama.py --model_config r1_distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
        --distill_config 80GB_distill_OpenR1_math \
        --eval_config eval_alpaca_clean \
        --lk_zero_init \
        --verbose --seed 0 --replicate 0
else
    # Use 40GB config for smaller GPUs
    echo "Using 40GB config"
    python distill_llama.py --model_config r1_distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
        --distill_config 40GB_distill_OpenR1_math \
        --eval_config eval_alpaca_clean \
        --lk_zero_init \
        --verbose --seed 0 --replicate 0
fi