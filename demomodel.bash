```bash
module load python \
module load cuda \
source .venv/bin/activate \
python -Wignore demo_lolcats_llm.py \
--attn_mlp_checkpoint_path 'checkpoints/r1_distill_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=40GB_distill_OpenR1_math-m=r1_distill_llama3_8b_lk_smd_wtk64_fd64_w01-f=None-s=0-se=0-re=0-lzi=1_distill_1000.pt' \
--num_generations 1 --benchmark
```