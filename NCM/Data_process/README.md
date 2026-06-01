# NCM Data Process

This folder converts the original NCM battery MAT files into sliding-window MAT files used by the diffusion workflow.

## Scripts

### `generate_train_sliding.py`

Generates full sliding-window data for the NCM train, validation, and test splits.

Source folders:

```text
NCM_1C/
NCM_2C/
NCM_3C/
```

These folders are for the original NCM MATLAB data. Put the required source files there:

```text
matlab_allcc.mat
matlab.mat
```

Run:

```bash
python generate_train_sliding.py
```

Diffusion staging outputs:

```text
data/*_train_sliding.mat
data/*_val_sliding.mat
data_for_generated/*_test_sliding.mat
```

`NCM_1C/`, `NCM_2C/`, and `NCM_3C/` are not the Diffusion input folders. They are the source-data folders. Keep original data there, then stage generated sliding files for Diffusion in `Data_process/data/` and `Data_process/data_for_generated/`.

The files in `Data_process/data/` are the files to place in:

```text
NCM/Diffusion/data/
```

Use these files for diffusion training, validation, and train/validation selected-window generation.

Use the staging folders below:

```text
Data_process/data/                  # train and val files for Diffusion
Data_process/data_for_generated/    # test files for Diffusion
```

Then copy the staging folders into Diffusion:

```text
NCM/Data_process/data/                 -> NCM/Diffusion/data/
NCM/Data_process/data_for_generated/   -> NCM/Diffusion/data_for_generated/
```

Use these files for low, middle, high, or random selected-window test generation.

## Folder Mapping

```text
NCM/Data_process/data/                 -> NCM/Diffusion/data/
NCM/Data_process/data_for_generated/   -> NCM/Diffusion/data_for_generated/
```

## Output MAT Fields

Generated sliding MAT files contain the fields expected by the Diffusion dataset loaders:

```text
GAF_cell
FRAG_cell
COND_VEC_cell
info_cell
SEG_idx_cell
segRanges_all
Uraw_cell
```
