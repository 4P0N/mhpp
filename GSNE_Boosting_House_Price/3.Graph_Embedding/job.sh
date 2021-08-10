#!/bin/sh
#
#SBATCH --job-name=graphpred
#SBATCH --output=housepred_log.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=1-12:00:00
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:p100:2
### Modules you like to load. For more details, see below("how to load modules")
module load gcc/6.4.0
module load openmpi/3.0.0
module load tensorflowgpu/1.6.0-python-3.6.4
module load xgboost/0.82-python-3.6.4
module load anaconda3/5.0.1
source activate ./envs
#module load pip
#pip install -r requirements.txt
#conda install --file requirements.txt
srun python ./Code/train.py 'cora_ml' 'glace'