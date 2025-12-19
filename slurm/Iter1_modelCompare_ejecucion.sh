#!/bin/bash

# Bucle for que genera una lista de números enteros del 1 al total_elementos
for ((i = 0; i < 18; i++))
do
    echo "Número: $i"
    sbatch --export=parametro1=$i,parametro2=MobileNetV2,parametro3=18_07_2024 run_iter1_deepLearning.sh
done
