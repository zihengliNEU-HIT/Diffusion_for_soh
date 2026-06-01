# LFP SOH Estimator

Image-only state-of-health (SOH) estimation for LFP battery capacity prediction. The model uses SOH-ready GAF MAT files produced by the Diffusion workflow.

## Project Structure

- `SOH_config.py` - central configuration and command-line arguments.
- `SOH_data_process.py` - MAT-file dataset loading and PyTorch dataloaders.
- `SOH_model.py` - configurable CNN encoder, transformer regressor, and model builder.
- `SOH_train.py` - training entry point with real/generated GAF dual-path loss.
- `SOH_test.py` - test-time inference, metrics, plots, and attention maps.
- `SOH_utils.py` - checkpoints, metrics, denormalization, and plotting helpers.
- `save_npy.py` - per-test-file inference that saves structured `.npy` outputs.

## Input From Diffusion

Keep the same `soh_input` layout as `LFP/Diffusion`:

```text
LFP/Diffusion/soh_input/train       -> LFP/soh/soh_input/train
LFP/Diffusion/soh_input/val         -> LFP/soh/soh_input/val
LFP/Diffusion/soh_input/test_low    -> LFP/soh/soh_input/test_low
LFP/Diffusion/soh_input/test_middle -> LFP/soh/soh_input/test_middle
LFP/Diffusion/soh_input/test_high   -> LFP/soh/soh_input/test_high
LFP/Diffusion/soh_input/test_random -> LFP/soh/soh_input/test_random
```

Expected files:

```text
*_train_soh.mat
*_val_soh.mat
*_test_soh.mat
```

Expected fields:

```text
GAF_real    # train files only
GAF_gen
CAPACITY
INFO
```

## Train

Use the root SOH input folder. The dataloader will read `train/`, `val/`, and `test_random/` under it.

```bash
python SOH_train.py --data_dir ./soh_input --output_dir ./train_output --checkpoint_dir ./soh_checkpoints
```

Training writes:

```text
train_output/loss_curve.png
soh_checkpoints/best_model.pt
soh_checkpoints/model_epoch_*.pt
soh_checkpoints/soh_min.npy
soh_checkpoints/soh_max.npy
```

## Export Test Results

Input directories keep the Diffusion names. Output directories add `_output` so generated results do not mix with input MAT files.

Default config:

```python
data_dir = "./soh_input/test_random"
output_dir = "./test_random_output"
checkpoint_dir = "./soh_checkpoints"
```

Run default random test export:

```bash
python save_npy.py
```

Run each test mode:

```bash
python save_npy.py --data_dir ./soh_input/test_low --output_dir ./test_low_output
python save_npy.py --data_dir ./soh_input/test_middle --output_dir ./test_middle_output
python save_npy.py --data_dir ./soh_input/test_high --output_dir ./test_high_output
python save_npy.py --data_dir ./soh_input/test_random --output_dir ./test_random_output
```

`SOH_test.py` is also available for aggregate metrics and attention maps:

```bash
python SOH_test.py --data_dir ./soh_input/test_random --output_dir ./test_random_output --checkpoint_dir ./soh_checkpoints
```

Test outputs include:

```text
metrics.txt
regression_plot.png
trajectory_battery*.png
predictions.npy
attn_maps/
```

`save_npy.py` also writes per-battery `.npy`, regression, and trajectory files.
