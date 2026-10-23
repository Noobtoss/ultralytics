#!/bin/bash
#SBATCH --job-name=ultralytics_arr # Kurzname des Jobs
#SBATCH --array=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --partition=p2,p3,p4,p5,p6 # p4
#SBATCH --qos=preemptible          # gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                  # Anzahl Knoten
#SBATCH --ntasks=1                 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4          # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=32G                  # RAM pro CPU Kern #20G #32G #64G

# ----- DIRS --------------------------------------------------------
ROOT=/nfs/scratch/staff/schmittth/code_nexus/ultralytics

# ----- GET ARGS ----------------------------------------------------
ARGS_FILE="$ROOT/custom/slurm/slurm_args.txt"
ARGS=$(grep -v '^[[:space:]]*#' "$ARGS_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID))p")

declare -A KV
read -r -a ARR <<< "$ARGS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done
[[ "$ARGS" != *"seed"* ]] && ARGS="$ARGS seed ${SLURM_ARRAY_JOB_ID}"

RUN_NAME="${KV[exp_name]:-unnamed_experiment}"
RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${ROOT}/runs/${RUN_NAME}"
MODEL="${KV[model]:-custom/cfg/cls_feat_yolo11x.yaml}"
CKPT="${KV[ckpt]:-checkpoints/yolo11x.pt}"
DATA="${KV[data]:-datasets/default.yaml}"

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3

eval "$(conda shell.bash hook)"
conda activate conda-ultralytics

export PYTHONPATH="$ROOT/custom/src:$PYTHONPATH"

# ----- WANDB -------------------------------------------------------
yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/nfs/scratch/staff/schmittth/tmp
export WANDB_CACHE_DIR=/nfs/scratch/staff/schmittth/tmp
export WANDB_CONFIG_DIR=/nfs/scratch/staff/schmittth/tmp

# ----- TRAINING ----------------------------------------------------
python $ROOT/custom/src/train.py \
       --run_name $RUN_NAME \
       --save_dir $OUT_DIR \
       --model    $ROOT/$MODEL \
       --ckpt     $ROOT/$CKPT \
       --data     $ROOT/$DATA  \
       $ARGS

# ----- CLEANUP -----------------------------------------------------
KEEP_FILES=("metrics.csv" "results.csv" "last.pt")

find "$OUT_DIR" -type f $(printf ' ! -name %s' "${KEEP_FILES[@]}") -delete
find "$OUT_DIR" -type d -empty -delete
