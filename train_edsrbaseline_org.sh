CUDA_VISIBLE_DEVICES=0 python3 main_org.py \
--data_test Set5 --dir_data <your_datasets_path> --n_GPUs 1 \
--scale 4 --model EDSR \
--cadyq --save edsrbaseline_org_x4 \
--n_feats 64 --n_resblocks 16 --res_scale 1 \
--patch_size 192 --batch_size 16 \
--epochs 600  --decay 300 --lr 1e-4 \
#