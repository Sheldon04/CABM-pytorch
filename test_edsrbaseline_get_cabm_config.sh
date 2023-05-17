CUDA_VISIBLE_DEVICES=0 python3 main_getconfig.py \
--test_only \
--data_test div2k_valid --dir_data <your_datasets_path> --n_GPUs 1 \
--scale 4 --k_bits 8 --model EDSR \
--cadyq --search_space 4+6+8 --save test_edsr_cadyq_patch \
--n_feats 64 --n_resblocks 16 --res_scale 1 \
--patch_size 96 --batch_size 16 --step_size 94 \
--student_weights <CADyQ_model_path> \
--test_patch \
# 