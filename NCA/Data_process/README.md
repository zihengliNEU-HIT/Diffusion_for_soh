# NCA Data Process

This folder converts the original NCA battery MAT files into sliding-window MAT files used by the diffusion workflow.

## Scripts

### `generate_dataset_sliding.py`

Generates full sliding-window data for the NCA train, validation, and test splits.

It reads `matlab.mat` and `matlab_128.mat` from each dataset folder:

```text
Dataset_1_NCA_batteryCY25-025_1/
Dataset_1_NCA_batteryCY25-05_1/
Dataset_1_NCA_batteryCY25-1_1/
```

Run:

```bash
python generate_dataset_sliding.py
```

Outputs:

```text
Dataset_1_NCA_batteryCY25-025_1/*_sliding.mat
Dataset_1_NCA_batteryCY25-05_1/*_sliding.mat
Dataset_1_NCA_batteryCY25-1_1/*_sliding.mat
data/*_sliding.mat
```

The copied files in `Data_process/data/` are the files to place in:

```text
NCA/Diffusion/data/
```

Use these files for diffusion training, validation, and train/validation selected-window generation.

### `generate_single_fragment_sliding.py`

Generates test-only fixed-fragment sliding-window data for selected voltage regions.

Supported fragment modes:

```text
low
middle
high
random
```

Run all modes:

```bash
python generate_single_fragment_sliding.py --all-modes
```

Run one mode:

```bash
python generate_single_fragment_sliding.py --segment-mode low
python generate_single_fragment_sliding.py --segment-mode middle
python generate_single_fragment_sliding.py --segment-mode high
python generate_single_fragment_sliding.py --segment-mode random
```

Outputs:

```text
data_for_generated/low/*_test_sliding.mat
data_for_generated/middle/*_test_sliding.mat
data_for_generated/high/*_test_sliding.mat
data_for_generated/random/*_test_sliding.mat
```

These files correspond to the test-generation input folder in Diffusion:

```text
NCA/Diffusion/data_for_generated/
```

For example, when running the random test workflow, copy the files from:

```text
NCA/Data_process/data_for_generated/random/
```

to:

```text
NCA/Diffusion/data_for_generated/
```

Then run:

```bash
python windowed_reverse_diffusion.py --data_dir ./data_for_generated --split test --test_window_mode random --output_dir generated_gaf/test_random
python soh_input_builder.py --real_dir ./data_for_generated --generated_dir generated_gaf/test_random --output_dir soh_input/test_random
```

## Folder Mapping

```text
NCA/Data_process/data/                         -> NCA/Diffusion/data/
NCA/Data_process/data_for_generated/<mode>/    -> NCA/Diffusion/data_for_generated/
```

`<mode>` should match the selected-window workflow you want to run: `low`, `middle`, `high`, or `random`.

## Output MAT Fields

Generated sliding MAT files contain the fields expected by the Diffusion dataset loaders:

```text
GAF_cell
FRAG_cell
COND_VEC_cell
info_cell
SEG_idx_cell
```
