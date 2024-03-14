#!/bin/bash


for i in $(seq 1 1 24)
do
    max_steps=10000
    for lr in -10 -11 -12 -13 -14 -15 -16 -8 -9
    do
        python siren_DT.py --experiment_name=$i --lr=$lr --sidelength=#512 --num_workers=16 --project=#project --max_steps=$max_steps --directory=#directory_for_images --batch_size=#18 --gpu_num=#0 --type=#origin
    done
done
