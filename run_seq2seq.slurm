#!/bin/bash
#SBATCH --job-name=seq2seq
#SBATCH --output=seq2seq_%j.out
#SBATCH --error=seq2seq_%j.out
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4

singularity exec --nv --bind /scratch/$USER \
--overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/MLLU-S22
python run_seq2seq.py $1
"