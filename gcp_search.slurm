#!/bin/bash
#SBATCH --job-name=DT
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G

# config
dataset=$1
delta_type=$2
data_ratio=$3

# logging dir
mkdir -p outputs/${dataset}/${delta_type}
log_dir=outputs/${dataset}/${delta_type}
log_path=${log_dir}/output_${data_ratio}.log

# Singularity path
ext3_path=/scratch/hl3797/overlay-25GB-500K.ext3
sif_path=/scratch/hl3797/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv --bind /scratch \
--overlay ${ext3_path}:ro ${sif_path} \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/MLLU-S22
python run_search.py ${dataset} ${delta_type} ${data_ratio} > ${log_path} 2>&1
"