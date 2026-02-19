# SuperMetal: a generative AI framework for rapid and precise metal ion location prediction in proteins

---

## Quick Start

```bash
# Install
git clone https://github.com/scofieldlinlin/SuperMetal.git
cd SuperMetal
conda create -n supermetal python=3.9 -y && conda activate supermetal
pip install -e .

# Run example
python predict.py --protein examples/example_protein.pdb --output results/
```

Output: `results/example_protein_combined.pdb` with protein + predicted zinc positions.

**Python API:**
```python
from predict import predict
results = predict("examples/example_protein.pdb")
print(results['cluster_centroids'])  # Predicted zinc coordinates
```

**Note:** First run downloads ESM weights (~2.5GB) and model checkpoints (~50MB). Subsequent runs use cache.

---

## Prediction

### Command Line
```bash
# Basic
python predict.py --protein protein.pdb

# With ground truth evaluation
python predict.py --protein protein.pdb --ground-truth ligands.mol2 --output results/
```

### Python API
```python
from predict import predict

results = predict("protein.pdb")
print(results['cluster_centroids'])  # Final predicted positions

# With evaluation
results = predict("protein.pdb", ground_truth_ligand_path="ligands.mol2")
print(f"Coverage: {results['metrics']['coverage']:.0f}%")
```

### Options
| Option | Default | Description |
|--------|---------|-------------|
| `--protein` | required | Input PDB file |
| `--output` | `.` | Output directory |
| `--num-metals` | 100 | Initial positions to generate |
| `--confidence-threshold` | 0.5 | Filtering threshold |
| `--cluster-eps` | 5.0 | DBSCAN clustering radius (Å) |
| `--no-confidence` | False | Disable confidence filtering |
| `--cpu` | False | Use CPU instead of GPU |

### Output Files
- `*_metal_predictions.pdb` - Predicted zinc positions only
- `*_combined.pdb` - Protein + predicted zinc positions

### Pre-trained Models
Models auto-download from [HuggingFace](https://huggingface.co/scofieldlinlin/SuperMetal) on first use.

Local checkpoints (if available):
- `workdir/large_all_atoms_model/best_model.pt` - Score model
- `workdir/large_confidence_model/best_model.pt` - Confidence model

---

## Training

Training has two stages: (1) Score model (diffusion) and (2) Confidence model (filtering).

### Option A: Download Pre-built Cache (Recommended)

Training data (preprocessed with ESM embeddings) is available on HuggingFace (~32GB):

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='scofieldlinlin/SuperMetal',
    allow_patterns='cache/*',
    local_dir='.'
)
"
```

### Option B: Process from Scratch

If you want to process your own PDB files:

```bash
# Step 1: Prepare FASTA from PDB files
python datasets/esm_embedding_preparation_metal.py \
  --data_dir your_pdb_folder/ \
  --out_file data/your_data.fasta

# Step 2: Generate ESM embeddings (~2.5GB model downloads on first run)
python esmfold/extract.py esm2_t33_650M_UR50D \
  data/your_data.fasta \
  data/your_embeddings/ \
  --repr_layers 33 --include per_tok --truncation_seq_length 4096

# Step 3: Train (cache auto-generated on first run)
python -m train \
  --data_dir your_pdb_folder/ \
  --split_train your_split.txt \
  --esm_embeddings_path data/your_embeddings/ \
  --cache_path data/your_cache/ \
  ...
```

### 1. Train Score Model (Diffusion)

```bash
python -m train \
  --cache_path cache \
  --split_train data/splits/train.txt \
  --split_val data/splits/val.txt \
  --log_dir workdir --run_name my_score_model \
  --n_epochs 500 --batch_size 8 \
  --ns 40 --nv 4 --num_conv_layers 3 \
  --distance_embed_dim 64 --cross_distance_embed_dim 64 \
  --all_atoms
```

Output: `workdir/my_score_model/best_model.pt`

### 2. Train Confidence Model

Requires a trained score model first.

```bash
python -m confidence.confidence_train \
  --original_model_dir workdir/my_score_model \
  --cache_path cache \
  --split_train data/splits/train.txt \
  --split_val data/splits/val.txt \
  --log_dir workdir --run_name my_confidence_model \
  --n_epochs 100 --batch_size 8 \
  --ns 24 --nv 6 --num_conv_layers 5 \
  --distance_embed_dim 32 --cross_distance_embed_dim 32 \
  --all_atoms
```

Output: `workdir/my_confidence_model/best_model.pt`

### Quick Test (smaller parameters)

For testing the training pipeline with limited GPU memory (uses included test data):

```bash
# Score model quick test (untested for accuracy)
python -m train \
  --data_dir data/zincbind_test_16 \
  --split_train data/splits/train_test16.txt \
  --split_val data/splits/train_test16.txt \
  --esm_embeddings_path data/embeddings_output_test \
  --log_dir workdir --run_name test_score_model \
  --n_epochs 10 --batch_size 2 --ns 8 --nv 2 --num_conv_layers 1 --all_atoms
```

**Note:** Smaller parameters are for testing only. Model quality is not guaranteed.

### Training Output
Saved to `workdir/<run_name>/`:
- `best_model.pt` - Best validation checkpoint
- `model_parameters.yml` - Hyperparameters

---

## Troubleshooting

### Wrong e3nn version
```bash
pip install e3nn==0.5.1  # Critical!
```

### Out of GPU memory
Reduce batch size: `--batch_size 2` or model size: `--ns 12 --nv 2`

### First run slow
Normal! Building cache takes 1-2 min for test data, 30-60 min for full dataset. Subsequent runs are instant.

---

## Citation

If you use SuperMetal in your research, please cite:

```bibtex
@article{lin2025supermetal,
  title={SuperMetal: a generative AI framework for rapid and precise metal ion location prediction in proteins},
  author={Lin, Xiaobo and Su, Zhaoqian and Liu, Yunchao and Liu, Jingxian and Kuang, Xiaohan and Cummings, Peter T and Spencer-Smith, Jesse and Meiler, Jens},
  journal={Journal of Cheminformatics},
  volume={17},
  number={1},
  pages={107},
  year={2025},
  publisher={Springer}
}
```

---

## Acknowledgments

Built upon [DiffDock](https://github.com/gcorso/DiffDock) and [ESM](https://github.com/facebookresearch/esm).

