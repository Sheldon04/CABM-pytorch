CUDA_VISIBLE_DEVICES=0 python3 main_inference.py \
--data_test Set5 --dir_data /root/data01/tsm/datasets --n_GPUs 1 \
--scale 4 --k_bits 8 --model EDSR \
--search_space 4+6+8 --save edsrbaseline_cabm_simple_x4 \
--n_feats 64 --n_resblocks 16 --res_scale 1 \
--patch_size 96 --batch_size 16 --test_patch --step_size 94 \
--epochs 300  --decay 150 --lr 1e-5 \
--loss_kd --loss_kdf --w_bit_decay 1e-6 --select_bit 0 --select_float 2 \
--teacher_weights <pams_model_path> \
--student_weights <pams_model_path> \
#