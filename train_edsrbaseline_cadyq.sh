CUDA_VISIBLE_DEVICES=0 python3 main_cadyq.py \
--data_test Set5 --dir_data <your_datasets_path> --n_GPUs 1 \
--scale 4 --k_bits 8 --model EDSR \
--cadyq --search_space 4+6+8 --save edsrbaseline_cadyq_x4 \
--n_feats 64 --n_resblocks 16 --res_scale 1 \
--patch_size 96 --batch_size 16 \
--epochs 300  --decay 150 --lr 1e-4 --bitsel_lr 1e-4 --bitsel_decay 150 \
--loss_kd --loss_kdf --w_bit 1e-4 --w_bit_decay 1e-6 \
--teacher_weights <pams_model_path> \
--student_weights <pams_model_path> \
#