#!/bin/bash
# Single GPU training script (12GB VRAM)
# Uses 16-sample test dataset for fast iteration
# ~1-2 minutes per epoch

python -m train \
--run_name my_model_12gb \
--all_atoms \
--test_sigma_intervals \
--esm_embeddings_path data/embeddings_output_test \
--data_dir data/zincbind_test_16 \
--split_train data/splits/train_test16.txt \
--split_val data/splits/val_test16.txt \
--cache_path data/cache_test \
--log_dir workdir \
--lr 1e-3 \
--tr_sigma_min 0.1 \
--tr_sigma_max 20 \
--dynamic_max_cross \
--batch_size 4 \
--ns 16 \
--nv 2 \
--num_conv_layers 2 \
--scheduler plateau \
--scale_by_sigma \
--dropout 0.1 \
--remove_hs \
--c_alpha_max_neighbors 24 \
--receptor_radius 15 \
--num_dataloader_workers 2 \
--num_workers 16 \
--val_inference_freq 5 \
--num_inference_complexes 8 \
--use_ema \
--distance_embed_dim 32 \
--cross_distance_embed_dim 32 \
--sigma_embed_dim 32 \
--scheduler_patience 100 \
--n_epochs 500 \
2>&1 | tee workdir/my_model_12gb/training.log
