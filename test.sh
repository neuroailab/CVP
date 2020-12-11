#!/bin/bash -ex
python3 write_train_test_txt.py

python3 test.py --dataset /mnt/fs4/fanyun/cvp_stimulis/$1,/mnt/fs4/fanyun/human_stimulis/$1 --gpu 1 --checkpoint ./output/tmp/vvn/$1rgb_cvp_traj_comb_late_fact_gc_n2e2n_Ppix1_iter10000.pth

python3 test.py --dataset /mnt/fs4/fanyun/cvp_stimulis/$1,/mnt/fs4/fanyun/cvp_stimulis/$1_val --gpu 1 --checkpoint ./output/tmp/vvn/$1rgb_cvp_traj_comb_late_fact_gc_n2e2n_Ppix1_iter10000.pth
