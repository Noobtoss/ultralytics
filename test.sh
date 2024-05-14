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

data=../semmelv2/datasets/semmel/testsets/testsetDIV2K/gray/images # ../datasets/semmel/yoloSetups/semmel33.yaml
cfg=cfgLAlu7.yaml
model=None
name=None
task=detect

while [ $# -gt 0 ]; do
  case "$1" in
    -d|-data|--data)       data="$2"   ;;
    -c|-cfg|--cfg)         cfg="$2"    ;;
    -m|-model|--model)     model="$2"  ;;
    -n|-name|--name)       name="$2"   ;;
    -t|-task|--task)       task="$2"   ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
  shift
done

: $data
: ${_%.*}
: $(basename $_)
run_name=$_

if [ $run_name == "." ]; then
     run_name=""
fi

: $model
: ${_%.*}
: $(basename $_)
: ${_,,}
: ${_^}
run_name=$run_name$_

if [ $name != "None" ]; then
     run_name=$run_name${name^}
fi

export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate yolo
pip install -r requirements.txt
pip install -qe . 

if [ $task == "train" ] || [ $task == "val" ] || [ $task == "test" ]; then
	srun yolo val cfg=$cfg mode=val data=$data project=runs name=$run_name/$task model=$model split=$task		
elif [ $task == "detect" ]; then
	srun yolo predict cfg=$cfg mode=predict source=$data project=runs name=$run_name/$task model=$model # conf=0.10 
else
 	echo "Unknown Task"
fi
