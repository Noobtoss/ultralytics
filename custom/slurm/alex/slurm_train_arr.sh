#!/bin/bash
#SBATCH --job-name=ultralytics_train_arr  # Name shown in squeue
#SBATCH --array=1-1%4       # Job array: tasks 1 to 1, max 4 running at once
#SBATCH --output=logs/R-%A-%a.out  # Log file: %A=jobID, %a=array task index
#SBATCH --gres=gpu:a100:2    # Request 2x A100 GPUs
#SBATCH --partition=a100     # Submit to the a100 node partition
#SBATCH --ntasks=1           # 1 process total (not MPI)
#SBATCH --ntasks-per-node=1  # That 1 process runs on 1 node
#SBATCH --cpus-per-task=4    # 4 CPU cores for that process (data loading etc)
#SBATCH --time=00:16:10      # Walltime limit: kill job after 16min 10sec
#SBATCH --mail-type=ALL      # Email on job start, end, fail
#SBATCH --mail-user=thomas.schmitt@th-nuernberg.de

# ----- BASE_DIR ----------------------------------------------------
BASE_DIR="$WORK/code_nexus/ultralytics"
DATA_DIR=$TMPDIR

# ----- GET ARGS ----------------------------------------------------
EXP_NAME="${KV[exp_name]:-unnamed_experiment}"
SAVE_DIR="${BASE_DIR}/runs/${EXP_NAME}"
MODEL="${KV[model]:-custom/cfg/cls_feats_yolo11x.yaml}"
CKPT="${KV[ckpt]:-checkpoints/yolo11x.pt}"
DATA="${KV[data]:-datasets/default.yaml}"

# ----- ENVIRONMENT SETUP -------------------------------------------
unset SLURM_EXPORT_ENV

module purge
module load python/3.12-conda
module load cuda/12.6.1

conda activate conda-ultralytics

# --- PROXY  --------------------------------------------------------
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# ----- WANDB -------------------------------------------------------
yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/tmp/ths_wandb      # Store wandb files on local /tmp
export WANDB_CACHE_DIR=/tmp/ths_wandb
export WANDB_CONFIG_DIR=/tmp/ths_wandb

# ----- ULTRALYTICS SETTINGS-----------------------------------------
yolo settings runs_dir="$BASE_DIR/runs"  # likely not needed
yolo settings weights_dir="$BASE_DIR/models"  # likely not needed

# ----- DATA STAGING ------------------------------------------------
PATH_TAR=$(grep "^path:" $DATA | cut -d ':' -f2 | xargs)
cp $DATA $DATA_DIR/
DATA="$DATA_DIR/$(basename $DATA)"
sed -i "s|^path:.*|path: $DATA_DIR|" $DATA

tar xf $PATH_TAR --strip-components=1 -C $DATA_DIR \
  --warning=no-unknown-keyword \
  --exclude='._*' \
  --exclude='.DS_Store' \
  --exclude='__MACOSX'

echo ErrorMessage unpacking: $?  # $? = exit code (0 = success, anything else = error)
echo $DATA
echo $DATA_DIR

# ----- TRAINING ----------------------------------------------------
python $BASE_DIR/custom/python/train.py \
       --exp_name $EXP_NAME \
       --save_dir $SAVE_DIR \
       --model    $BASE_DIR/$MODEL \
       --ckpt     $BASE_DIR/$CKPT \
       --data     $BASE_DIR/$DATA  \
       $PARAMS

# ----- CLEANUP -----------------------------------------------------
KEEP_FILES=("metrics.csv" "results.csv" "last.pt")
eval find $SAVE_DIR -type f $(printf ' ! -name "%s"' "${KEEP_FILES[@]}") -delete
find $SAVE_DIR -type d -empty -delete
