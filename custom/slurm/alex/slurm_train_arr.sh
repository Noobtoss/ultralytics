#!/bin/bash
#SBATCH --job-name=ultralytics_train_arr  # Name shown in squeue
#SBATCH --array=99-129%8         # 87-98 # 1-86 Job array: tasks 1 to 23, max 8 running at once
#SBATCH --output=logs/R-%A-%a.out  # Log file: %A=jobID, %a=array task index
#SBATCH --gres=gpu:a40:1     # Request 1x A40 GPUs
#SBATCH --partition=a40      # Submit to the a40 node partition
#SBATCH --ntasks=1           # 1 process total (not MPI)
#SBATCH --ntasks-per-node=1  # That 1 process runs on 1 node
#SBATCH --cpus-per-task=4    # 4 CPU cores for that process (data loading etc)
#SBATCH --time=02:32:32      # Walltime limit: kill job after 2hr 32min 32sec
#SBATCH --mail-type=ALL      # Email on job start, end, fail
#SBATCH --mail-user=thomas.schmitt@th-nuernberg.de

# ----- BASE_DIR ----------------------------------------------------
BASE_DIR="$WORK/code_nexus/ultralytics"
JOB_DIR=$TMPDIR

# ----- GET ARGS ----------------------------------------------------
PARAMS_FILE="$BASE_DIR/custom/slurm/alex/slurm_params.txt"
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
SAVE_DIR="${BASE_DIR}/runs/${EXP_NAME}"
MODEL="${KV[model]:-custom/cfg/cls_feat_yolo11x.yaml}"
CKPT="${KV[ckpt]:-checkpoints/yolo11x.pt}"
DATA="${KV[data]:-datasets/default.yaml}"

# ----- ENVIRONMENT SETUP -------------------------------------------
unset SLURM_EXPORT_ENV

module purge
module load python/3.12-conda
module load cuda/12.6.1

conda activate conda-ultralytics

export PYTHONPATH="$BASE_DIR/custom/python"  # "$BASE_DIR/custom/python:$PYTHONPATH"

# --- PROXY  --------------------------------------------------------
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# ----- WANDB -------------------------------------------------------
yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/tmp/ths_wandb
export WANDB_CACHE_DIR=/tmp/ths_wandb
export WANDB_CONFIG_DIR=/tmp/ths_wandb

# ----- ULTRALYTICS SETTINGS-----------------------------------------
# yolo settings datasets_dir=$JOB_DIR
# yolo settings runs_dir="$BASE_DIR/runs"
# yolo settings weights_dir="$BASE_DIR/models"

# ----- DATA STAGING ------------------------------------------------
PATH_TAR=$(grep "^path:" $BASE_DIR/$DATA | cut -d ':' -f2 | xargs)
tar xf $PATH_TAR --strip-components=1 -C $JOB_DIR \
  --warning=no-unknown-keyword \
  --exclude='._*' \
  --exclude='.DS_Store' \
  --exclude='__MACOSX'

echo ErrorMessage unpacking: $?  # $? = exit code (0 = success, anything else = error)

cp $BASE_DIR/$DATA $JOB_DIR/
DATA="$JOB_DIR/$(basename $DATA)"
sed -i "s|^path:.*|path: $JOB_DIR|" $DATA
PARAMS=$(echo "$PARAMS" | sed "s|data [^ ]*|data $DATA|")
echo $DATA
echo $JOB_DIR

# ----- TRAINING ----------------------------------------------------
python $BASE_DIR/custom/python/train.py \
       --exp_name $EXP_NAME \
       --save_dir $SAVE_DIR \
       --model    $BASE_DIR/$MODEL \
       --ckpt     $BASE_DIR/$CKPT \
       --data     $DATA  \
       $PARAMS

# ----- CLEANUP -----------------------------------------------------
KEEP_FILES=("metrics.csv" "results.csv" "last.pt")
eval find $SAVE_DIR -type f $(printf ' ! -name "%s"' "${KEEP_FILES[@]}") -delete
find $SAVE_DIR -type d -empty -delete
