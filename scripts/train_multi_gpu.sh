#!/bin/bash
# Multi-GPU training (Original paper configuration)
# Requires multiple GPUs with ≥40GB total VRAM

python -m train \
--run_name large_model_multi_gpu \
--all_atoms \
--test_sigma_intervals \
--esm_embeddings_path data/embeddings_output_cleaned \
--data_dir data/zincbind_cleaned_processed \
--split_train data/splits/train.txt \
--split_val data/splits/val.txt \
--split_test data/splits/test_metal3d.txt \
--log_dir workdir \
--lr 1e-3 \
--tr_sigma_min 0.1 \
--tr_sigma_max 20 \
--dynamic_max_cross \
--batch_size 8 \
--ns 40 \
--nv 4 \
--num_conv_layers 3 \
--scheduler plateau \
--scale_by_sigma \
--dropout 0.1 \
--remove_hs \
--c_alpha_max_neighbors 24 \
--receptor_radius 15 \
--num_dataloader_workers 2 \
--num_workers 16 \
--cudnn_benchmark \
--val_inference_freq 20 \
--num_inference_complexes 500 \
--use_ema \
--distance_embed_dim 64 \
--cross_distance_embed_dim 64 \
--sigma_embed_dim 64 \
--scheduler_patience 100 \
--n_epochs 500 \
--wandb \
2>&1 | tee workdir/large_model_multi_gpu/training.log

