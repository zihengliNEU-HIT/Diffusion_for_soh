# NCM Data Process

This folder contains the Python version of the NCM sliding-window MATLAB workflow.

## Scripts

- `generate_train_sliding.py` reproduces the `code1_26convec.m` workflow in `NCM_1C`, `NCM_2C`, and `NCM_3C`.
- The script loads only `matlab_allcc.mat` and `matlab.mat` from each NCM dataset folder.
- Generated `*_sliding.mat` files are written back to their matching dataset folders.
- Generated `*_train_sliding.mat` files are also copied into `data/`.

## Usage

```bash
python generate_train_sliding.py
```

## Outputs

Each saved MAT file contains:

- `GAF_cell`
- `FRAG_cell`
- `SEG_idx_cell`
- `info_cell`
- `segRanges_all`
- `Uraw_cell`
- `COND_VEC_cell`

`COND_VEC_cell` stores the 26-dimensional condition vector in the same order as the original MATLAB scripts.
