#!/bin/bash
#SBATCH --job-name=DT
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

# fetch config
method=$1
dataset=$2
curr_time=$(date +"%Y%m%d_%H%M%S")
config_path=configs/${method}/${dataset}.json

# init logging dir
log_dir=log/${method}/${dataset}
log_path=${log_dir}/output_${curr_time}.log
mkdir -p log/${method}/${dataset}
cp ${config_path} ${log_dir}/config_${curr_time}.json

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv --bind /scratch/$USER \
--overlay ${ext3_path}:ro ${sif_path} \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/MLLU-S22
python run_seq2seq.py ${config_path} ${log_path} > ${log_path} 2>&1
"