TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 2222 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ~/multimodal/TSP3D-main/ \
    --val_freq 3 --batch_size 6 --save_freq 6 --print_freq 500 \
    --lr=5e-4 --keep_trans_lr=5e-4 --voxel_size=0.01 --num_workers=16 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate \
    --log_dir ~/multimodal/TSP3D-main/output/logs/test \
    --lr_decay_epochs 50 75 \
    --augment_det \
    --checkpoint_path ~/multimodal/TSP3D-main/checkpoints/ckpt_epoch_91.pth \
    --eval