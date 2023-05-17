CUDA_VISIBLE_DEVICES=0 python3 main_inference.py \
--test_only \
--data_test Urban100 --dir_data /home/notebook/data/group/tsm/datasets --n_GPUs 1 \
--scale 4 --k_bits 8 --model EDSR \
--search_space 4+6+8 --save edsrbaseline_test_x4 \
--n_feats 64 --n_resblocks 16 --res_scale 1 \
--patch_size 96 --select_bit 0 --select_float 2 --calibration 100 \
--student_weights ./experiment/edsrbaseline_edge_1e-5_1b2b_x4/model/model_best.pth.tar \
--test_patch --step_size 96 \
# 