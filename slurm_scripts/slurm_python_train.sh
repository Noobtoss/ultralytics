#!/bin/bash
#SBATCH --job-name=yolo_array    # Kurzname des Jobs
#SBATCH --array=1-20%3            # 3 Jobs total running 2 at a time
#SBATCH --output=logs/R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=32G                # RAM pro CPU Kern #20G #32G #64G

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda_ultralytics

yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/nfs/scratch/staff/schmittth/.cache

BASE_DIR=/nfs/scratch/staff/schmittth/sync/ultralytics

python $BASE_DIR/train_scripts/python_train.py --index $SLURM_ARRAY_TASK_ID
