# NCM Data Process

This folder converts the original NCM battery MAT files into sliding-window MAT files used by the diffusion workflow.

## Scripts

### `generate_train_sliding.py`

Generates full sliding-window data for the NCM train, validation, and test splits.

It reads `matlab_allcc.mat` and `matlab.mat` from each dataset folder:

```text
NCM_1C/
NCM_2C/
NCM_3C/
```

Run:

```bash
python generate_train_sliding.py
```

Outputs:

```text
NCM_1C/*_sliding.mat
NCM_2C/*_sliding.mat
NCM_3C/*_sliding.mat
data/*_train_sliding.mat
```

`NCM_1C/`, `NCM_2C/`, and `NCM_3C/` are the original dataset folders. The script reads the original MAT files from these folders and writes generated `*_sliding.mat` files back beside the original data. It also copies generated train files into `Data_process/data/`.

The copied train files in `Data_process/data/` are the files to place in:

```text
NCM/Diffusion/data/
```

Use these files for diffusion training and train selected-window generation.

If you also need validation and test files for Diffusion, copy the generated validation/test `*_sliding.mat` files from the original dataset folders into the staging folders below:

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
