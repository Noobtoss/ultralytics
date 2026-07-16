#!/bin/bash
#SBATCH --job-name=ultralytics_train_arr # Kurzname des Jobs
#SBATCH --array=1%8
#SBATCH --output=logs/R-%A-%a.out
#SBATCH --partition=p2,p4,p6        # p1,p2,p3,p4,p5,p6
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

# ----- ROOT_DIR ----------------------------------------------------
ROOT_DIR=/nfs/scratch/staff/schmittth/code_nexus/ultralytics

# ----- GET ARGS ----------------------------------------------------
PARAMS_FILE="$ROOT_DIR/custom/slurm/slurm_params.txt"
PARAMS=$(grep -v '^[[:space:]]*#' "$PARAMS_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID))p")

PARAMS=$(echo "$PARAMS" | sed -E "s/(exp_name[[:space:]]+[^[:space:]]+)/\1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/")
declare -A KV
read -r -a ARR <<< "$PARAMS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done
[[ "$PARAMS" != *"seed"* ]] && PARAMS="$PARAMS seed ${SLURM_ARRAY_JOB_ID}"

EXP_NAME="${KV[exp_name]:-unnamed_experiment}"
SAVE_DIR="${ROOT_DIR}/runs/${EXP_NAME}"
MODEL="${KV[model]:-custom/cfg/cls_feat_yolo11x.yaml}"
CKPT="${KV[ckpt]:-checkpoints/yolo11x.pt}"
DATA="${KV[data]:-datasets/default.yaml}"

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
module load cuda/cuda-11.8.0
eval "$(conda shell.bash hook)"

conda activate conda-ultralytics

export PYTHONPATH="$ROOT_DIR/custom/python"  # "$ROOT_DIR/custom/python:$PYTHONPATH"

export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/ultralytics_${SLURM_JOB_ID}_XXXXXX")

# ----- WANDB -------------------------------------------------------
yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- TRAINING ----------------------------------------------------
python $ROOT_DIR/custom/python/train.py \
       --exp_name $EXP_NAME \
       --save_dir $SAVE_DIR \
       --model    $ROOT_DIR/$MODEL \
       --ckpt     $ROOT_DIR/$CKPT \
       --data     $ROOT_DIR/$DATA  \
       $PARAMS

# ----- CLEANUP -----------------------------------------------------
wandb sync --sync-all || true
rm -rf "$TMPDIR"
KEEP_FILES=("metrics.csv" "results.csv" "last.pt")
rm -rf "$SAVE_DIR/wandb"
eval find $SAVE_DIR -type f $(printf ' ! -name "%s"' "${KEEP_FILES[@]}") -delete
find $SAVE_DIR -type d -empty -delete
