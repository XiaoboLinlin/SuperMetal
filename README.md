
# SuperMetal: A Generative AI Framework for Rapid and Precise Metal Ion Location Prediction in Proteins

SuperMetal is a state-of-the-art generative AI framework designed to predict metal ion locations within proteins with high precision. This framework builds upon [DiffDock](https://github.com/gcorso/DiffDock) and introduces modifications to simultaneously diffuse multiple metal ions over 3D space. SuperMetal integrates a confidence model and clustering mechanism to improve prediction accuracy.

## Features
- Predicts metal ion binding sites in protein structures
- Uses 3D diffusion-based generative modeling
- Enhanced accuracy through a confidence model and clustering
- Supports various metal ions, including zinc

## Setup and Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-repo/SuperMetal.git
   cd SuperMetal
   ```

2. **Install Requirements**  
   Ensure you have a Python environment set up, then install the necessary packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**  
   Place your dataset in the `data/` directory, structured as outlined in the dataset preparation step below.

## Steps to Retrain the Model

### Step 1: Prepare FASTA Data
Prepare the data for embedding:
```bash
python datasets/esm_embedding_preparation_metal.py \
--data_dir data/zincbind_cleaned_processed \
--out_file data/prepared_for_esm_metal_zincbind_cleaned.fasta
```

### Step 2: Train the Model
Run the main training script:
```bash
python -m train \
--run_name large_all_atoms_model \
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
--num_workers 2 \
--wandb \
--cudnn_benchmark \
--val_inference_freq 20 \
--num_inference_complexes 500 \
--use_ema \
--distance_embed_dim 64 \
--cross_distance_embed_dim 64 \
--sigma_embed_dim 64 \
--scheduler_patience 100 \
--n_epochs 500
```

### Step 3: Train the Confidence Model
Run the following script to train the confidence model:
```bash
python -m confidence.confidence_train \
--original_model_dir workdir/large_all_atoms_model \
--data_dir data/zincbind_cleaned_processed \
--all_atoms \
--run_name large_confidence_model \
--cache_path data/large_cache_confidence \
--split_train data/splits/train.txt \
--split_val data/splits/val.txt \
--split_test data/splits/test_metal3d.txt \
--inference_steps 20 \
--samples_per_complex 1 \
--batch_size 8 \
--batch_size_preprocessing 1 \
--n_epochs 100 \
--wandb \
--lr 1e-3 \
--scheduler_patience 50 \
--ns 24 \
--nv 6 \
--num_conv_layers 5 \
--dynamic_max_cross \
--scale_by_sigma \
--dropout 0.1 \
--remove_hs \
--c_alpha_max_neighbors 24 \
--receptor_radius 15 \
--esm_embeddings_path data/esm2_3billion_embeddings.pt \
--main_metric confidence_loss \
--main_metric_goal min \
--best_model_save_frequency 5 \
--rmsd_classification_cutoff 5 \
--cache_creation_id 1 \
--cache_ids_to_combine 1
```

### Step 4: Run Evaluation
To evaluate the model, run:
```bash
python -m validation_matrix.validation_1 \
--original_model_dir workdir/all_atoms_model \
--confidence_dir workdir/confidence_model \
--split_test data/splits/test_noMetal3d_noOverlap.txt \
--batch_size_preprocessing 1 \
--rmsd_classification_cutoff 5 \
--prob_cutoff 0.5
```

### Step 5: Speed Test and Visualization
Run the speed test and visualization script:
```bash
python speedTest/speed_test.py
```

## License
This project is licensed under the MIT License.