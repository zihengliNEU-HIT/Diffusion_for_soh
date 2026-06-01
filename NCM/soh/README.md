# NCM SOH Estimator

Image-only state-of-health (SOH) estimation for NCM battery capacity
prediction. The model uses Gramian Angular Field (GAF) images as input, a
configurable CNN encoder, and a lightweight self-attention regressor.

## Project Structure

- `SOH_config.py` - central configuration and command-line arguments.
- `SOH_data_process.py` - MAT-file dataset loading and PyTorch dataloaders.
- `SOH_model.py` - configurable CNN encoder, transformer regressor, and model builder.
- `SOH_train.py` - training entry point with real/generated GAF dual-path loss.
- `SOH_test.py` - test-time inference, metrics, plots, and attention maps.
- `SOH_utils.py` - checkpoints, metrics, denormalization, and plotting helpers.
- `save_npy.py` - per-test-file inference that saves structured `.npy` outputs.

## Data Format

The code expects split-specific MAT files in `data_dir`:

- Train files: `*_train_soh.mat`
- Validation files: `*_val_soh.mat`
- Test files: `*_test_soh.mat`

Expected fields:

- `GAF_real`: `[N, H, W]`, used by training files.
- `GAF_gen`: `[N, K, H, W]`, generated GAF images.
- `CAPACITY`: `[N, 1]`, capacity values in Ah.
- `INFO`: `[N, 2]`, battery id and cycle index.

`CAPACITY` is min-max normalized using the training split. The scaler is saved
to `soh_min.npy` and `soh_max.npy` under `checkpoint_dir`.

## Quick Start

Install the Python dependencies used by the scripts:

```bash
pip install numpy scipy matplotlib torch
```

Train the model. Use `./soh_data` as the training data directory. Checkpoints
are saved to `checkpoint_dir`, and the training loss curve is saved to
`output_dir`.

```bash
python SOH_train.py --data_dir ./soh_data --output_dir ./train_output --checkpoint_dir ./soh_checkpoints
```

For test-time export, `save_npy.py` is recommended instead of `SOH_test.py`
because it saves one structured `.npy` file and plots for each test MAT file.
Before running it, edit `SOH_config.py` and set the test input/output pair you
want:

```python
data_dir = "./random"
output_dir = "./test_random"
checkpoint_dir = "./soh_checkpoints"
```

Other common test sets use the same pattern:

```python
data_dir = "./middle"
output_dir = "./test_middle"

data_dir = "./low"
output_dir = "./test_low"

data_dir = "./high"
output_dir = "./test_high"
```

Then run:

```bash
python save_npy.py
```

`SOH_test.py` is still available for metrics, `predictions.npy`, and attention
map visualization:

```bash
python SOH_test.py --data_dir ./random --output_dir ./test_random --checkpoint_dir ./soh_checkpoints
```

## Outputs

Training saves checkpoints and `loss_curve.png`. Testing saves:

- `metrics.txt`
- `regression_plot.png`
- `trajectory_battery*.png`
- `predictions.npy`
- `attn_maps/` when attention visualization is enabled

`save_npy.py` saves per-battery structured arrays with `gt`, `pred`, `bat_idx`,
and `cycle_idx` fields, plus per-battery regression and trajectory plots.
