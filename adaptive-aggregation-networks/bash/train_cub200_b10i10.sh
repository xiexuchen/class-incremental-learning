#!/bin/bash
PYTHON='/public/home/daijiahao/anaconda3/bin/python'
for seed in 32 40 50; do
    for incre in 10; do
        ${PYTHON} ../main.py --nb_cl_fg=${incre} --nb_cl=${incre} \
        --gpu=0 --random_seed=${seed} --baseline="lucir" \
        --branch_mode="dual" --branch_1="ss" --branch_2="free" \
        --dataset="cub200" --epochs=90 --image_size=224 --disable_gpu_occupancy \
        --num_classes=190 --K=1 --the_lambda=10 --dist=0.5 --custom_weight_decay=0.0005
    done
done