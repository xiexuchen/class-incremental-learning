#!/bin/bash
PYTHON='/public/home/daijiahao/anaconda3/bin/python'

for seed in 32 40 50; do
    for init_cls in 2; do
        ${PYTHON} ../main.py --nb_cl_fg=${init_cls} --nb_cl=${init_cls} \
        --gpu=0 --random_seed=${seed} --baseline="lucir" \
        --branch_mode="dual" --branch_1="ss" --branch_2="free" \
        --dataset="cifar100" --epochs=160 --image_size=224 --disable_gpu_occupancy \
        --nb_protos=10
    done
done