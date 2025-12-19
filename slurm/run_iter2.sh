#!/bin/bash
#SBATCH --job-name=DT_iter2
#SBATCH -o outputs/dtIter2_%j.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e errors/dtIter2_%j.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:2
#SBATCH -c 64
#SBATCH --time 6:00:00
#SBATCH --mem-per-cpu=3G

module load cesga/2020 gcccore/system tensorflow/2.5.0-cuda-system
python3 ./telegram_bot.py -m "Start deeptrack GPU Iter2" -p $parametro1
python3 ./Iter2_quality.py -c $parametro1 -t $parametro2
python3 ./telegram_bot.py -m "Stop deeptrack GPU Iter2" -p $parametro1
