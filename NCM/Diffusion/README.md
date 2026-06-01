# NCM Conditional CAF Diffusion and SOH Input Preparation

This repository contains the NCM version of the conditional CAF/GAF diffusion workflow. It trains and evaluates a diffusion model, generates selected-window GAF samples, and converts those generated MAT files into SOH-estimation inputs.

This README is written for the current NCM codebase. The NCM-specific settings are:

```text
chemistry labels      : 1C / 2C / 3C
condition vector dim  : 26
default timesteps     : 400
window workflow       : low / middle / high / random
```

`SOH_estimate` is not included here yet.

## Project Structure

```text
.
|-- cs_main.py                         # Main entry point for diffusion training/testing
|-- train.py                           # Diffusion training loop
|-- test.py                            # Diffusion testing loop
|-- reverse_diffusion.py               # Standard reverse diffusion generation
|-- windowed_reverse_diffusion.py      # Selected-window GAF generation for SOH preparation
|-- soh_input_builder.py               # Converts generated GAF files into SOH-ready MAT files
|-- dataset.py                         # Standard MAT dataset
|-- windowed_dataset.py                # Selected-window MAT dataset
|-- unet.py                            # Conditional U-Net
|-- diffusion_model.py                 # EMA wrapper and sampling interface
|-- gaussian_diffusion.py              # DDPM/DDIM utilities
|-- visualization.py                   # Figure export helpers
|-- configs/
|   |-- cs_main_train.json
|   |-- cs_main_test.json
|   |-- reverse_diffusion.json
|   |-- windowed_reverse_diffusion.json
|   `-- soh_input_builder.json
|-- data/                              # Train/validation/test sliding MAT files
|-- data_for_generated/                # Held-out test-generation sliding MAT files
|-- checkpoints/                       # Checkpoints and condition-vector statistics
|-- generated_data/                    # Standard reverse diffusion output
|-- generated_gaf/                     # Selected-window reverse diffusion output
|-- matlab/                            # Capacity/SOH source MAT files
|-- soh_input/                         # SOH-ready MAT output
|-- output/                            # Training/testing figures and metrics
|-- requirements.txt
`-- README.md
```

## Installation

Create a Python 3.10+ environment and install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch manually if your environment requires a specific CUDA build.

## Data Folders

### `data/`

Put train/validation/test sliding MAT files here. This folder is used for diffusion training/testing, selected-window train/validation generation, and SOH input construction.

Expected file names follow this pattern:

```text
1Cbattery1_02_train_sliding.mat
1Cbattery2_02_val_sliding.mat
1Cbattery3_02_test_sliding.mat
2Cbattery1_02_train_sliding.mat
3Cbattery1_02_test_sliding.mat
```

Required MAT fields:

```text
GAF_cell
FRAG_cell
COND_VEC_cell
info_cell
SEG_idx_cell    # optional
```

### `data_for_generated/`

Put held-out test sliding MAT files here if you want generation-only test outputs. Test selected-window generation reads this folder in the recommended commands.

### `checkpoints/`

Put trained model checkpoints and condition-vector normalization files here:

```text
model_epoch_140.pt
model_best.pt
cond_mean.npy
cond_std.npy
```

`cond_mean.npy` and `cond_std.npy` are required by reverse generation.

### `matlab/`

Put NCM capacity/SOH source files here:

```text
matlab1C.mat
matlab2C.mat
matlab3C.mat
```

These paths can be changed in `configs/soh_input_builder.json`.

## Workflow

### 1. Train the Diffusion Model

```bash
python cs_main.py --config configs/cs_main_train.json
```

Main outputs:

```text
checkpoints/model_epoch_*.pt
checkpoints/model_best.pt
checkpoints/cond_mean.npy
checkpoints/cond_std.npy
output/training_metrics.png
output/training_data.txt
```

### 2. Test the Diffusion Model

```bash
python cs_main.py --config configs/cs_main_test.json
```

Set `checkpoint_path` in `configs/cs_main_test.json` before testing.

Main outputs:

```text
output/test_metrics.txt
output/test_samples.png
```

### 3. Optional Standard Reverse Diffusion

```bash
python reverse_diffusion.py --config configs/reverse_diffusion.json
```

This is the standard generation workflow and writes generated MAT files and analysis figures to `generated_data/`.

### 4. Selected-Window GAF Generation

Use this workflow when preparing GAF inputs for SOH estimation.

`windowed_dataset.py` first removes empty windows and missing condition vectors, then selects windows using NCM rules:

```text
train: low + middle + high windows per cycle
val  : one selected window, default random
test : one selected window, choose low / middle / high / random
```

For NCM, if a cycle has `n_valid_windows` valid windows:

```text
low    = min(3, n_valid_windows - 1)
high   = max(n_valid_windows - 3, 0)
middle = (low + high) // 2
```

If a cycle has too few valid windows, these positions naturally fall back to an available window.

#### Train Generation

Train generation reads from `./data`. Each cycle keeps three windows. With `num_samples=20`, each cycle gets:

```text
3 windows x 20 generated images = 60 generated GAF images
```

Run:

```bash
python windowed_reverse_diffusion.py   --config configs/windowed_reverse_diffusion.json   --data_dir ./data   --split train   --num_samples 20   --output_dir generated_gaf/train
```

Window-level output:

```text
GAF_gen  [N*3, 20, H, W]
INFO     [N*3, 2]
```

`soh_input_builder.py` later groups the three windows from the same cycle into:

```text
GAF_gen  [N, 60, H, W]
```

#### Validation Generation

Validation generation also reads from `./data`. Each cycle uses one random valid window and generates 20 images:

```bash
python windowed_reverse_diffusion.py   --config configs/windowed_reverse_diffusion.json   --data_dir ./data   --split val   --val_window_mode random   --num_samples 20   --output_dir generated_gaf/val
```

Output:

```text
GAF_gen  [N, 20, H, W]
INFO     [N, 2]
```

#### Test Generation

Test generation reads from `./data_for_generated`. Generate four selected-window variants, each with 5 generated images per cycle:

```bash
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data_for_generated --split test --test_window_mode low --num_samples 5 --output_dir generated_gaf/test_low
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data_for_generated --split test --test_window_mode middle --num_samples 5 --output_dir generated_gaf/test_middle
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data_for_generated --split test --test_window_mode high --num_samples 5 --output_dir generated_gaf/test_high
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data_for_generated --split test --test_window_mode random --num_samples 5 --output_dir generated_gaf/test_random
```

Each test output folder contains:

```text
GAF_gen  [N, 5, H, W]
INFO     [N, 2]
```

### 5. Build SOH Input MAT Files

After each selected-window generation command finishes, run the SOH builder for the corresponding generated folder. `real_dir` points to the sliding MAT folder that matches the selected split and window mode.

```bash
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data --generated_dir generated_gaf/train --output_dir soh_input/train
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data --generated_dir generated_gaf/val --output_dir soh_input/val
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data_for_generated --generated_dir generated_gaf/test_low --output_dir soh_input/test_low
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data_for_generated --generated_dir generated_gaf/test_middle --output_dir soh_input/test_middle
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data_for_generated --generated_dir generated_gaf/test_high --output_dir soh_input/test_high
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data_for_generated --generated_dir generated_gaf/test_random --output_dir soh_input/test_random
```

Folder mapping:

```text
generated_gaf/train       -> soh_input/train
generated_gaf/val         -> soh_input/val
generated_gaf/test_low    -> soh_input/test_low
generated_gaf/test_middle -> soh_input/test_middle
generated_gaf/test_high   -> soh_input/test_high
generated_gaf/test_random -> soh_input/test_random
```

SOH input fields:

```text
GAF_gen
CAPACITY
INFO
GAF_real    # train split only
```

## Configuration Files

All commands use JSON config files in `configs/`. Command-line values can override config values.

Important NCM fields:

```text
cond_vector_dim: 26
timesteps      : 400
chemistry      : 1C / 2C / 3C
```

`configs/windowed_reverse_diffusion.json` controls selected-window generation:

```text
data_dir
output_dir
checkpoint_dir
checkpoint_path
split
num_samples
val_window_mode
test_window_mode
window_seed
```

`configs/soh_input_builder.json` controls SOH input construction:

```text
data_dir
matlab_sources
expected_train_windows
compare_splits
real_dir
generated_dir
output_dir
```

## Recommended Run Order

```bash
python cs_main.py --config configs/cs_main_train.json
python cs_main.py --config configs/cs_main_test.json
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data --split train --num_samples 20 --output_dir generated_gaf/train
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data --split val --val_window_mode random --num_samples 20 --output_dir generated_gaf/val
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data_for_generated --split test --test_window_mode low --num_samples 5 --output_dir generated_gaf/test_low
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data_for_generated --split test --test_window_mode middle --num_samples 5 --output_dir generated_gaf/test_middle
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data_for_generated --split test --test_window_mode high --num_samples 5 --output_dir generated_gaf/test_high
python windowed_reverse_diffusion.py --config configs/windowed_reverse_diffusion.json --data_dir ./data_for_generated --split test --test_window_mode random --num_samples 5 --output_dir generated_gaf/test_random
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data --generated_dir generated_gaf/train --output_dir soh_input/train
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data --generated_dir generated_gaf/val --output_dir soh_input/val
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data_for_generated --generated_dir generated_gaf/test_low --output_dir soh_input/test_low
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data_for_generated --generated_dir generated_gaf/test_middle --output_dir soh_input/test_middle
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data_for_generated --generated_dir generated_gaf/test_high --output_dir soh_input/test_high
python soh_input_builder.py --config configs/soh_input_builder.json --real_dir ./data_for_generated --generated_dir generated_gaf/test_random --output_dir soh_input/test_random
```
