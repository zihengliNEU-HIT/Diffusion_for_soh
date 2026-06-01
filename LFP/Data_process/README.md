# LFP Data Process

This folder converts the original LFP battery MAT files into sliding-window MAT files used by the diffusion workflow.

## Scripts

### `generate_train_sliding.py`

Generates full sliding-window data for the LFP train, validation, and test splits.

It reads the original resampled MAT file from each dataset folder:

```text
5_6C_19/5_6C_19PER_4_6C_NEWSTRUCTURE_resampled.mat
5_6C_36/5_6C_36PER_4_3C_NEWSTRUCTURE_resampled.mat
5_3C_54/5_3C_54PER_4C_NEWSTRUCTURE_resampled.mat
```

Run:

```bash
python generate_train_sliding.py
```

The dataset folders are the source-data folders. The script also writes generated `*_sliding.mat` files back beside the matching original MAT file for traceability.

Diffusion-ready staging outputs:

```text
data/*_train_sliding.mat
data/*_val_sliding.mat
data_for_generated/*_test_sliding.mat
```

The copied files in `Data_process/data/` are the files to place in:

```text
LFP/Diffusion/data/
```

Use these files for diffusion training and validation.

The copied files in `Data_process/data_for_generated/` are the files to place in:

```text
LFP/Diffusion/data_for_generated/
```

Use these files for test selected-window generation, such as `test_low`, `test_middle`, `test_high`, or `test_random`.

## Folder Mapping

```text
LFP/Data_process/data/               -> LFP/Diffusion/data/
LFP/Data_process/data_for_generated/ -> LFP/Diffusion/data_for_generated/
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
```
