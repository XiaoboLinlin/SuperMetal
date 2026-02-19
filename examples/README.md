# Example Files

This folder contains example files to test SuperMetal predictions.

## Quick Test

```bash
python predict.py --protein examples/example_protein.pdb --output results/
```

## More Test Proteins

Additional test proteins are available in `data/zincbind_test_16/`:

| Protein | Zinc Sites | Files |
|---------|------------|-------|
| 1Z3A | 2 | `data/zincbind_test_16/1Z3A/` |
| 3BUD | 1 | `data/zincbind_test_16/3BUD/` |
| 1NS3 | 1 | `data/zincbind_test_16/1NS3/` |
| 5ZYA | 3 | `data/zincbind_test_16/5ZYA/` |
| 2LXH, 3DGV, 3M4G, 3MA2, 3V0B, 4BZ5, 4LGR, 4LNB, 4U5G, 5IY9, 5M8M, 6FUL | varies | `data/zincbind_test_16/` |

### Run with ground truth evaluation
```bash
# Test on 1Z3A (2 zinc sites)
python predict.py \
  --protein data/zincbind_test_16/1Z3A/1Z3A_protein_processed.pdb \
  --ground-truth data/zincbind_test_16/1Z3A/1Z3A_ligands.mol2

# Test on 5ZYA (3 zinc sites)
python predict.py \
  --protein data/zincbind_test_16/5ZYA/5ZYA_protein_processed.pdb \
  --ground-truth data/zincbind_test_16/5ZYA/5ZYA_ligands.mol2
```

### Python API
```python
from predict import predict

# Test multiple proteins
test_cases = ['1Z3A', '3BUD', '1NS3', '5ZYA']
for pdb in test_cases:
    results = predict(
        f"data/zincbind_test_16/{pdb}/{pdb}_protein_processed.pdb",
        ground_truth_ligand_path=f"data/zincbind_test_16/{pdb}/{pdb}_ligands.mol2"
    )
    print(f"{pdb}: Coverage={results['metrics']['coverage']:.0f}%, Precision={results['metrics']['precision']:.0f}%")
```
