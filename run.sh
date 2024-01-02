#!/bin/bash
#SBATCH --comment=cifar10
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=xl.wang@sheffield.ac.uk

# Load the conda module
module load Anaconda3/2022.10
# Load cuda
module load cuDNN/8.4.1.50-CUDA-11.7.0
# 进入conda环境
source activate similarity
# 切换目录
cd /users/acq21xw/similarity
# 安装依赖环境
pip install -r requirements.txt
pip install wandb
wandb login
pip install avalanche-lib

# 执行以下工作流程
#python run2.1.py > log.txt 2>&1
python download_data.py
python generate_txt.py
python run1.1.py
