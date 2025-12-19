#!/bin/bash

# Bucle for que genera una lista de números enteros del 1 al total_elementos
for ((i = 0; i < 6; i++))
do
    echo "Número: $i"
    sbatch --export=parametro1=$i,parametro2=31_07_2024 run_iter2.sh
done
