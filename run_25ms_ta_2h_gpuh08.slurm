#!/usr/bin/env bash
#SBATCH --job-name=mert_25ms_ta_2h
#SBATCH --partition=NV_H100
#SBATCH --account=cit_ylju
#SBATCH -w gpuh08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAH10080GBHBM3:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/25ms_ta_2h_gpuh08_%j.out
#SBATCH --error=logs/25ms_ta_2h_gpuh08_%j.err

set -euo pipefail

cd /public_hw/home/cit_ylju/yinlin/cft研究/code/ssl_cft/mert25ms其他改进/mert_cft-main_baseline_baseline_25ms_baseline
mkdir -p logs

source /public_hw/software/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate /public_hw/home/cit_ylju/conda_envs/mert_old_py312_cu121

export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/public_hw/home/cit_ylju/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import sys
import torch
import transformers

print("python", sys.executable, flush=True)
print("python_version", sys.version.split()[0], flush=True)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
print("transformers", transformers.__version__, flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_device_count", torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), flush=True)
PY

python train_conp_v6_0415.py --config config_mert_base.yaml
