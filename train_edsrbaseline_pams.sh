CUDA_VISIBLE_DEVICES=0 python3 main_pams.py \
--data_test Set5 --dir_data <your_datasets_path> --n_GPUs 1 \
--scale 4 --k_bits 8 --model EDSR \
--save edsrbaseline_pams_x4 \
--n_feats 64 --n_resblocks 16 --res_scale 1 \
--patch_size 96 --batch_size 16 \
--epochs 100 --lr 1e-5 --decay 50 \
--teacher_weights <fullprecision_model_path> \
--student_weights <fullprecision_model_path> \
#