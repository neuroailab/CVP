#!/bin/bash -ex
python3 write_train_test_txt.py
python3 train.py --dataset /mnt/fs4/fanyun/cvp_stimulis/collide2_new --gpu 3,4
