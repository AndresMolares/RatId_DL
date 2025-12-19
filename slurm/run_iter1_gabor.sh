#!/bin/bash
#SBATCH --job-name=DT_gabor
#SBATCH -o outputs/dtGabor_%j.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e errors/dtGabor_%j.e       # Name of stderr output file(%j expands to jobId)
#SBATCH -C clk
#SBATCH --qos=clk_medium
#SBATCH --array=0-29
#SBATCH --time 3-00:00:00
#SBATCH --mem-per-cpu=32G

module load cesga/system tensorflow/2.11.0
python3 ./telegram_bot.py -m "Start deeptrack GABOR" -p $SLURM_ARRAY_TASK_ID
python3 ./Iter1_modelCompare_GABOR.py -c $SLURM_ARRAY_TASK_ID -t "19_03_2025_Gabor"
python3 ./telegram_bot.py -m "Stop deeptrack GABOR" -p $SLURM_ARRAY_TASK_ID
