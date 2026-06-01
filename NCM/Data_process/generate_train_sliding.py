#!/usr/bin/env python3
"""Generate NCM sliding-window MAT files from the original MATLAB workflow."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parent
IMG_SIZE = 128
SEGMENT_SIZE = 0.1
OVERLAP = 0.5
EFFECTIVE_STEP = SEGMENT_SIZE * (1 - OVERLAP)
V_BOT = 3.7
V_TOP = 4.1


@dataclass(frozen=True)
class DatasetConfig:
    # Static settings copied from each NCM code1_26convec.m script.
    name: str
    prefix: str
    protocol_id: int
    nbat_limit: int
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    test_ids: tuple[int, ...]


DATASETS = (
    # NCM 1C split: train=[1,3,5,8], val=[4,6], test=[2,7].
    DatasetConfig("NCM_1C", "1C", 1, 8, (1, 3, 5, 8), (4, 6), (2, 7)),
    # NCM 2C split: train=[1,3,5,8], val=[4,6], test=[2,7].
    DatasetConfig("NCM_2C", "2C", 2, 8, (1, 3, 5, 8), (4, 6), (2, 7)),
    # NCM 3C split: train=[1,2,4,7,8,9,11,12,14,15], val=[6,10], test=[3,5,13].
    DatasetConfig("NCM_3C", "3C", 3, 15, (1, 2, 4, 7, 8, 9, 11, 12, 14, 15), (6, 10), (3, 5, 13)),
)


def matlab_struct_array(value: np.ndarray) -> list:
    # Flatten MATLAB struct arrays in column-major order.
    return list(np.asarray(value, dtype=object).ravel(order="F"))


def cycles_of(battery_item) -> list:
    # Return the cycle struct list for one battery.
    return matlab_struct_array(battery_item.cycles)


def as_vector(value) -> np.ndarray:
    # Convert MATLAB row/column arrays to one-dimensional numeric vectors.
    return np.asarray(value, dtype=np.float64).reshape(-1)


def as_scalar(value) -> float:
    # Convert MATLAB scalar arrays to Python floats.
    return float(np.asarray(value).squeeze())


def load_batteries(dataset_dir: Path) -> tuple[list, list]:
    # Match MATLAB: matlab_allcc.mat is full-cycle data, matlab.mat is fixed-cycle data.
    full_data = loadmat(dataset_dir / "matlab_allcc.mat", squeeze_me=False, struct_as_record=False)
    fix_data = loadmat(dataset_dir / "matlab.mat", squeeze_me=False, struct_as_record=False)
    return matlab_struct_array(full_data["battery"]), matlab_struct_array(fix_data["battery"])


def interp_to_size(values: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    # Reproduce MATLAB interp1(..., 'linear') when the input length is not IMG_SIZE.
    values = as_vector(values)
    if values.size == size:
        return values
    x_old = np.arange(1, values.size + 1, dtype=np.float64)
    x_new = np.linspace(1, values.size, size)
    return np.interp(x_new, x_old, values)


def mat2gray(values: np.ndarray) -> np.ndarray:
    # Match MATLAB mat2gray normalization for each GAF matrix.
    values = np.asarray(values, dtype=np.float64)
    v_min = np.min(values)
    v_max = np.max(values)
    if v_max <= v_min:
        return np.zeros_like(values, dtype=np.float64)
    return (values - v_min) / (v_max - v_min)


def to_uint8_image(values: np.ndarray) -> np.ndarray:
    # Match MATLAB uint8(255 * mat2gray(...)).
    return np.clip(np.floor(255 * mat2gray(values) + 0.5), 0, 255).astype(np.uint8)


def colon(start: float, step: float, stop: float) -> np.ndarray:
    # MATLAB-style start:step:stop range generation.
    values = []
    current = start
    eps = abs(step) * 1e-10
    while current <= stop + eps:
        values.append(current)
        current += step
    return np.asarray(values, dtype=np.float64)


def empty_cell(shape: tuple[int, int]) -> np.ndarray:
    # Create a NumPy object array that savemat writes as a MATLAB cell array.
    cell = np.empty(shape, dtype=object)
    cell.fill(np.empty((0, 0), dtype=np.float64))
    return cell


def split_name_for_battery(battery_id: int, config: DatasetConfig) -> str:
    # Use the same train/val/test split rules as the MATLAB scripts.
    if battery_id in config.train_ids:
        return "train"
    if battery_id in config.val_ids:
        return "val"
    if battery_id in config.test_ids:
        return "test"
    raise ValueError(f"Battery {battery_id} is not assigned to train/val/test")


def make_cond_vec(cycle, previous_cycle, cycle_id: int, lo: float, hi: float, useg: np.ndarray,
                  useg_rs: np.ndarray, mask: np.ndarray, protocol_id: int) -> np.ndarray:
    # Build the 26-dimensional condition vector in the original MATLAB order.
    vmin_n = lo
    vmax_n = hi
    width = (vmax_n - vmin_n) * protocol_id
    mid = (vmin_n + vmax_n) / 2
    cycle_num = np.float32(cycle_id)
    cycle_sqrt = np.float32(np.sqrt(cycle_id))
    cycle_log = np.float32(np.log(1 + cycle_id))
    vmax_diff_3p7 = np.float32(hi - 4.2)
    v_mean = float(np.mean(useg_rs))
    v_std = float(np.std(useg_rs, ddof=1))
    slope = float((useg_rs[-1] - useg_rs[0]) / useg_rs.size)
    diff_mid_vs_segmean = mid - v_mean
    v_energy = float(np.trapezoid(useg_rs) / useg_rs.size)
    dv = np.diff(useg_rs)
    max_slope = float(np.max(dv) * protocol_id)
    min_slope = float(np.min(dv) * protocol_id)
    chg_val = np.float32(as_scalar(cycle.charge_rate))
    dsc_val = np.float32(as_scalar(cycle.discharge_rate))
    prev_dsc = np.float32(as_scalar(previous_cycle.discharge_rate)) if previous_cycle is not None else np.float32(1)
    t_full = as_vector(cycle.relative_time_min)
    i_full = as_vector(cycle.current_A)
    t_seg = t_full[mask]
    i_seg = i_full[mask]
    if t_seg.size == 0:
        t_seg = np.asarray([t_full[0], t_full[-1]], dtype=np.float64)
        i_seg = np.asarray([i_full[0], i_full[-1]], dtype=np.float64)
    elif t_seg.size == 1:
        t_seg = np.asarray([t_seg[0], t_seg[0] + 1e-3], dtype=np.float64)
        i_seg = np.asarray([i_seg[0], i_seg[0]], dtype=np.float64)
    if t_seg.size > 1:
        dt_hours = np.diff(t_seg) / 60
        i_avg = (i_seg[:-1] + i_seg[1:]) / 2
        seg_capacity = float(np.sum(np.abs(i_avg * dt_hours)))
    else:
        seg_capacity = 0.0
    v_seg_orig = useg[: min(useg.size, i_seg.size)]
    i_seg_orig = i_seg[: v_seg_orig.size]
    seg_avg_power = float(np.mean(np.abs(v_seg_orig * i_seg_orig)))
    seg_power_density = seg_avg_power / seg_capacity if seg_capacity > 1e-9 else 0.0
    t_mean = float(np.mean(t_seg))
    t_duration = float(np.max(t_seg) - np.min(t_seg))
    seg_capacity_ln = float(np.log(1 + seg_capacity))
    seg_capacity_rate = seg_capacity / max(t_duration, 1e-9)
    cond_vec = np.asarray(
        [
            chg_val,
            dsc_val,
            prev_dsc,
            cycle_num,
            cycle_sqrt,
            cycle_log,
            protocol_id,
            vmin_n,
            vmax_n,
            width,
            mid,
            v_mean,
            v_std,
            slope,
            diff_mid_vs_segmean,
            v_energy,
            max_slope,
            min_slope,
            vmax_diff_3p7,
            t_mean,
            t_duration,
            seg_capacity,
            seg_avg_power,
            seg_power_density,
            seg_capacity_ln,
            seg_capacity_rate,
        ],
        dtype=np.float32,
    )
    return cond_vec.reshape(1, -1)


def process_battery(config: DatasetConfig, battery_id: int, battery_full: list, battery_fix: list) -> dict:
    # Generate all MATLAB output variables for one battery.
    full_cycles = cycles_of(battery_full[battery_id - 1])
    fix_cycles = cycles_of(battery_fix[battery_id - 1])
    n_cycles = min(len(full_cycles), len(fix_cycles))
    gaf_values, frag_rows, seg_idx_rows, info_values, seg_ranges_values, cond_rows, uraw_values = [], [], [], [], [], [], []
    for cycle_index in range(1, n_cycles + 1):
        cycle = full_cycles[cycle_index - 1]
        fix_cycle = fix_cycles[cycle_index - 1]
        uraw = as_vector(cycle.voltage_V)
        uraw_values.append(uraw.reshape(-1, 1))
        ufixed_rs = interp_to_size(as_vector(fix_cycle.voltage_V))
        u01 = np.clip((ufixed_rs - V_BOT) / (V_TOP - V_BOT), 0, 1)
        theta_u = u01 * np.pi
        full_gaf = np.cos(theta_u[:, None] + theta_u[None, :])
        gaf_values.append(to_uint8_image(full_gaf))
        info_values.append(np.asarray([[battery_id, cycle_index]], dtype=np.float64))
        starts = colon(float(np.min(uraw)), EFFECTIVE_STEP, float(np.max(uraw) - SEGMENT_SIZE))
        if starts.size == 0:
            starts = np.asarray([float(np.min(uraw))], dtype=np.float64)
        if starts[-1] + SEGMENT_SIZE < float(np.max(uraw)) - 1e-9:
            starts = np.append(starts, float(np.max(uraw) - SEGMENT_SIZE))
        ranges = np.column_stack((starts, np.minimum(starts + SEGMENT_SIZE, float(np.max(uraw)))))
        frag_items, idx_items, cond_items, valid_ranges = [], [], [], []
        for lo, hi in ranges:
            mask = (uraw >= lo) & (uraw <= hi)
            useg = uraw[mask]
            if useg.size == 0:
                useg = np.asarray([lo, hi], dtype=np.float64)
            elif useg.size == 1:
                useg = np.asarray([useg[0], useg[0] + max(1e-6, hi - lo)], dtype=np.float64)
            useg_rs = interp_to_size(useg)
            useg01 = np.clip((useg_rs - lo) / max(1e-9, hi - lo), 0, 1)
            theta_s = useg01 * np.pi
            frag_gaf = np.cos(theta_s[:, None] + theta_s[None, :])
            previous_cycle = full_cycles[cycle_index - 2] if cycle_index > 1 else None
            cond_vec = make_cond_vec(cycle, previous_cycle, cycle_index, lo, hi, useg, useg_rs, mask, config.protocol_id)
            frag_items.append(to_uint8_image(frag_gaf))
            idx_items.append(np.asarray([[len(frag_items)]], dtype=np.float64))
            cond_items.append(cond_vec)
            valid_ranges.append([lo, hi])
        frag_rows.append(frag_items)
        seg_idx_rows.append(idx_items)
        cond_rows.append(cond_items)
        seg_ranges_values.append(np.asarray(valid_ranges, dtype=np.float64))
    rows = len(gaf_values)
    max_cols = max(len(row) for row in frag_rows) if frag_rows else 0
    gaf_cell = empty_cell((rows, 1))
    info_cell = empty_cell((rows, 1))
    seg_ranges_all = empty_cell((rows, 1))
    uraw_cell = empty_cell((rows, 1))
    frag_cell = empty_cell((rows, max_cols))
    seg_idx_cell = empty_cell((rows, max_cols))
    cond_vec_cell = empty_cell((rows, max_cols))
    for row in range(rows):
        gaf_cell[row, 0] = gaf_values[row]
        info_cell[row, 0] = info_values[row]
        seg_ranges_all[row, 0] = seg_ranges_values[row]
        uraw_cell[row, 0] = uraw_values[row]
        for col, value in enumerate(frag_rows[row]):
            frag_cell[row, col] = value
            seg_idx_cell[row, col] = seg_idx_rows[row][col]
            cond_vec_cell[row, col] = cond_rows[row][col]
    return {
        "GAF_cell": gaf_cell,
        "FRAG_cell": frag_cell,
        "SEG_idx_cell": seg_idx_cell,
        "info_cell": info_cell,
        "segRanges_all": seg_ranges_all,
        "Uraw_cell": uraw_cell,
        "COND_VEC_cell": cond_vec_cell,
    }


def main() -> None:
    # Write each dataset output in place and mirror Diffusion input folders.
    data_dir = ROOT / "data"
    data_for_generated_dir = ROOT / "data_for_generated"
    data_dir.mkdir(exist_ok=True)
    data_for_generated_dir.mkdir(exist_ok=True)
    for config in DATASETS:
        dataset_dir = ROOT / config.name
        battery_full, battery_fix = load_batteries(dataset_dir)
        nbat = min(config.nbat_limit, len(battery_full), len(battery_fix))
        for battery_id in range(1, nbat + 1):
            output = process_battery(config, battery_id, battery_full, battery_fix)
            split_name = split_name_for_battery(battery_id, config)
            filename = f"{config.prefix}battery{battery_id}_02_{split_name}_sliding.mat"
            output_path = dataset_dir / filename
            savemat(output_path, output, do_compression=False)
            if split_name in {"train", "val"}:
                shutil.copy2(output_path, data_dir / filename)
            elif split_name == "test":
                shutil.copy2(output_path, data_for_generated_dir / filename)
            print(f"Saved -> {output_path.relative_to(ROOT)}")
    print("Done. Train/val MAT files were copied to data/ and test MAT files to data_for_generated/.")


if __name__ == "__main__":
    main()
