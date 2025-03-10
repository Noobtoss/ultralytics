#!/bin/bash
#SBATCH --job-name=yolo        # Kurzname des Jobs
#SBATCH --output=logs/T-%j.out
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
CONFIG=$1
DATA=$2
MODEL=$3
TASK="test" # "val" # "train" # "detect"
# RUN_NAME="$BASE_DIR/runs/$(basename "${CONFIG%.*}")-$(basename "${DATA%.*}" | tr '[:upper:]' '[:lower:]')-$SLURM_JOB_ID"
PROJECT="runs"
NAME="$(basename "${CONFIG%.*}")-$(basename "${DATA%.*}" | tr '[:upper:]' '[:lower:]')-${SEED}-${SLURM_JOB_ID}"

srun yolo val cfg=$BASE_DIR/$CONFIG mode=val data=$BASE_DIR/$DATA project=$PROJECT name=$NAME model=$BASE_DIR/$MODEL split=$TASK
# if TASK=="detect"
# srun yolo predict cfg=$BASE_DIR/$CONFIG mode=predict source=$BASE_DIR/$DATA project=$PROJECT name=$NAME model=$BASE_DIR/$MODEL
