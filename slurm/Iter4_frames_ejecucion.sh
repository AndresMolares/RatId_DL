#!/bin/bash

# Bucle for que genera una lista de números enteros del 1 al total_elementos
for ((i = 0; i < 33; i++))
do
    echo "Número: $i"
    sbatch --export=parametro1=$i,parametro2=13_08_2024 run_iter4.sh
done
