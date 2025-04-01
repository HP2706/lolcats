module load python 
source .venv/bin/activate || uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

module load gcc/11.3.0 || module load gcc/12.1.0 || module load gcc/12.2.0 || module load gcc/12.3.0

cd ./csrc/ && python setup.py install 
uv pip install flash-attn --no-build-isolation