#!/usr/bin/env bash

for i in 0 1 2 3 4
do
    for m in 0.0 0.3 0.5 0.7
    do
        for n in 1 2 3 4
        do
            for h in 16 32 64 128
            do
            python mujoco.py --h_channels $h --hh_channels $h --layers $n --lr 0.001 --method "euler" --missing_rate $m --time_seq 50 --y_seq 10 --intensity '' --epoch 500 --step_mode 'valloss' --model dualdynamics_x_n
            python mujoco.py --h_channels $h --hh_channels $h --layers $n --lr 0.001 --method "euler" --missing_rate $m --time_seq 50 --y_seq 10 --intensity '' --epoch 500 --step_mode 'valloss' --model dualdynamics_x_g
            python mujoco.py --h_channels $h --hh_channels $h --layers $n --lr 0.001 --method "euler" --missing_rate $m --time_seq 50 --y_seq 10 --intensity '' --epoch 500 --step_mode 'valloss' --model dualdynamics_x_r
            python mujoco.py --h_channels $h --hh_channels $h --layers $n --lr 0.001 --method "euler" --missing_rate $m --time_seq 50 --y_seq 10 --intensity '' --epoch 500 --step_mode 'valloss' --model dualdynamics_x_c
            done
        done
    done
done

