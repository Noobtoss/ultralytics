#!/bin/bash
#SBATCH --job-name=yolo        # Kurzname des Jobs
#SBATCH --output=T-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58

conda activate ultralytics

BASE_DIR=/nfs/scratch/staff/schmittth/sync/ultralytics
CONFIG=$1
DATA=$2
MODEL=$3
TASK="test" # "val" # "train" # "detect"
RUN_NAME="$BASE_DIR/runs/$(basename "${CONFIG%.*}")-$(basename "${DATA%.*}" | tr '[:upper:]' '[:lower:]')-$SLURM_JOB_ID"
echo $RUN_NAME

srun yolo val cfg=$BASE_DIR/$CONFIG mode=val data=$BASE_DIR/$DATA project=runs name=$RUN_NAME/$TASK model=$BASE_DIR/$MODEL split=$TASK
# if TASK=="detect" srun yolo predict cfg=$CONFIG mode=predict source=$DATA project=runs name=$RUN_NAME/$TASK model=$MODEL