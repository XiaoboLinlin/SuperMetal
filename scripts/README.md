# Training Scripts

Ready-to-use training commands for SuperMetal.

## Single GPU (12GB)
```bash
bash train_single_gpu_12gb.sh
```
Fast iteration with 16-sample test dataset. ~1-2 minutes per epoch.

## Multi-GPU (40GB+)
```bash
bash train_multi_gpu.sh
```
Full dataset (9,387 samples). Original paper configuration.

---

## Customize

Copy and modify:
```bash
cp train_single_gpu_12gb.sh my_custom_training.sh
# Edit parameters, dataset, etc.
bash my_custom_training.sh
```

## Output Logs

Logs saved to `workdir/<run_name>/training.log`

View:
```bash
tail -f workdir/my_model_12gb/training.log
```

