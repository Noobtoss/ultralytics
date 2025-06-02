#!/bin/bash
#SBATCH --job-name=yolo_array    # Kurzname des Jobs
#SBATCH --array=1-32%2            # 3 Jobs total running 2 at a time
#SBATCH --output=logs/R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=32G                # RAM pro CPU Kern #20G #32G #64G

mkdir -p logs

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda_ultralytics

yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/tmp/ths_wandb
export WANDB_CACHE_DIR=/tmp/ths_wandb
export WANDB_CONFIG_DIR=/tmp/ths_wandb
# /nfs/scratch/staff/schmittth/.cache

BASE_DIR=/nfs/scratch/staff/schmittth/codeNexus/ultralytics

wait_time=$(((SLURM_ARRAY_TASK_ID - 1) * 2 * 60))  # This multiplies job ID by 60 to get seconds
echo "Waiting for $wait_time seconds ((SLURM_ARRAY_TASK_ID -1) * 4 * 60)"
sleep $wait_time

python $BASE_DIR/python_scripts/train_arr.py --index $SLURM_ARRAY_TASK_ID
