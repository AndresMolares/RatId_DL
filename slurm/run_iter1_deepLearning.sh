#!/bin/bash
#SBATCH --job-name=DT_GPU_iter1_${$parametro1}
#SBATCH -o outputs/dtGPU_iter1_%j.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e errors/dtGPU_iter1_%j.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:2
#SBATCH -c 64
#SBATCH --time 06:00:00
#SBATCH --mem-per-cpu=3G

module load cesga/2020 gcccore/system tensorflow/2.5.0-cuda-system
python3 ./Iter1_modelCompare_DEEPLEARNING.py -c $parametro1 -m $parametro2 -t $parametro3
