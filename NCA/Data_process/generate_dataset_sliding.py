#!/usr/bin/env python3
"""Generate sliding-window MAT files for the three NCA datasets."""

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
U_MIN = 3.0
U_MAX = 4.2


@dataclass(frozen=True)
class DatasetConfig:
    # Static settings copied from each original MATLAB sliding-window script.
    name: str
    prefix: str
    nca: int
    train_set: tuple[int, ...]
    val_set: tuple[int, ...]
    test_set: tuple[int, ...]


DATASETS = (
    # 1C split: train=[1,3,5,7,8], val=[2,9], test=[6,4].
    DatasetConfig("Dataset_1_NCA_batteryCY25-1_1", "1C", 3, (1, 3, 5, 7, 8), (2, 9), (6, 4)),
    # 0.25C split: train=[1,3,5,7,4], val=[2], test=[6].
    DatasetConfig("Dataset_1_NCA_batteryCY25-025_1", "025C", 1, (1, 3, 5, 7, 4), (2,), (6,)),
    # 0.5C split: train=[1,3,5,6,7,9,11,13,15,12,17,18], val=[2,10,14,19], test=[4,8,16].
    DatasetConfig(
        "Dataset_1_NCA_batteryCY25-05_1",
        "05C",
        2,
        (1, 3, 5, 6, 7, 9, 11, 13, 15, 12, 17, 18),
        (2, 10, 14, 19),
        (4, 8, 16),
    ),
)


def matlab_struct_array(value: np.ndarray) -> list:
    # Flatten MATLAB struct arrays using MATLAB column-major order.
    return list(np.asarray(value, dtype=object).ravel(order="F"))


def cycles_of(battery_item) -> list:
    # Return the cycle struct list for one battery.
    return matlab_struct_array(battery_item.cycles)


def as_vector(value) -> np.ndarray:
    # Convert MATLAB row/column arrays to a one-dimensional numeric vector.
    return np.asarray(value, dtype=np.float64).reshape(-1)


def as_scalar(value) -> float:
    # Convert MATLAB scalar arrays to a Python float.
    return float(np.asarray(value).squeeze())


def reset_cycle_time(batteries: list) -> None:
    # Match the MATLAB preprocessing step: each cycle starts at time zero.
    for battery in batteries:
        for cycle in cycles_of(battery):
            if hasattr(cycle, "time_s"):
                time_s = as_vector(cycle.time_s)
                cycle.time_s = time_s - time_s[0]


def load_batteries(dataset_dir: Path) -> tuple[list, list]:
    # Load the full-resolution and fixed-window MATLAB battery structs.
    full_data = loadmat(dataset_dir / "matlab.mat", squeeze_me=False, struct_as_record=False)
    fix_data = loadmat(dataset_dir / "matlab_128.mat", squeeze_me=False, struct_as_record=False)
    battery_full = matlab_struct_array(full_data["battery"])
    battery_fix = matlab_struct_array(fix_data["battery"])
    reset_cycle_time(battery_full)
    reset_cycle_time(battery_fix)
    return battery_full, battery_fix


def interp_to_size(values: np.ndarray, size: int = IMG_SIZE) -> tuple[np.ndarray, bool]:
    # Reproduce interp1(..., 'linear') when the input is not already IMG_SIZE.
    values = as_vector(values)
    if values.size == size:
        return values, False
    x_old = np.arange(1, values.size + 1, dtype=np.float64)
    x_new = np.linspace(1, values.size, size)
    return np.interp(x_new, x_old, values), True


def matlab_gaf(theta_t: np.ndarray, theta_u: np.ndarray, interpolated: bool) -> np.ndarray:
    # Preserve MATLAB row/column orientation after interp1 for GAF construction.
    if interpolated:
        return np.cos(theta_t[None, :] + theta_u[:, None])
    return np.cos(theta_t[:, None] + theta_u[None, :])


def mat2gray(values: np.ndarray) -> np.ndarray:
    # Match MATLAB mat2gray normalization for each GAF matrix.
    values = np.asarray(values, dtype=np.float64)
    v_min = np.min(values)
    v_max = np.max(values)
    if v_max <= v_min:
        return np.zeros_like(values, dtype=np.float64)
    return (values - v_min) / (v_max - v_min)


def to_uint8_image(values: np.ndarray) -> np.ndarray:
    # Match uint8(255 * mat2gray(...)) conversion.
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


def dataset_split_name(battery_id: int, config: DatasetConfig) -> str:
    # Use the same train/val/test filename rules as the MATLAB scripts.
    if battery_id in config.train_set:
        split = "train"
    elif battery_id in config.val_set:
        split = "val"
    elif battery_id in config.test_set:
        split = "test"
    else:
        split = "other"
    return f"{config.prefix}battery{battery_id}_02_{split}_sliding.mat"


def empty_cell(shape: tuple[int, int]) -> np.ndarray:
    # Create a NumPy object array that savemat writes as a MATLAB cell array.
    cell = np.empty(shape, dtype=object)
    cell.fill(np.empty((0, 0), dtype=np.float64))
    return cell


def make_cond_vec(cycle, previous_cycle, cycle_id: int, lo: float, hi: float, useg_rs: np.ndarray,
                  tseg_rs: np.ndarray, iseg: np.ndarray, tseg: np.ndarray, useg: np.ndarray,
                  protocol_id: int) -> np.ndarray:
    # Build the 22-dimensional condition vector in the original MATLAB order.
    chg_val = np.float32(as_scalar(cycle.charge_rate))
    dsc_val = np.float32(as_scalar(cycle.discharge_rate))
    prev_dsc = np.float32(as_scalar(previous_cycle.discharge_rate)) if previous_cycle is not None else np.float32(1)
    cycle_num = np.float32(cycle_id)
    cycle_sqrt = np.float32(np.sqrt(cycle_id))
    cycle_log = np.float32(np.log(1 + cycle_id))
    del prev_dsc, cycle_num, cycle_sqrt, cycle_log
    if tseg.size > 1:
        dt_hours = np.diff(tseg) / 3600
        i_avg = (iseg[:-1] + iseg[1:]) / 2
        seg_capacity = float(np.sum(np.abs(i_avg * dt_hours)))
    else:
        seg_capacity = 0.0
    seg_avg_power = float(np.mean(np.abs(useg * iseg)))
    seg_power_density = seg_avg_power / seg_capacity if seg_capacity > 1e-9 else 0.0
    seg_capacity_ln = float(np.log(1 + seg_capacity))
    seg_duration = float(np.max(tseg) - np.min(tseg))
    seg_capacity_rate = seg_capacity / seg_duration if seg_duration > 1e-9 else 0.0
    # These values are computed to mirror MATLAB, but the final 22-D vector does not include them.
    del seg_power_density, seg_capacity_ln, seg_capacity_rate
    vmin_n = np.float32(lo)
    vmax_n = np.float32(hi)
    mid = np.float32((vmin_n + vmax_n) / 2)
    umean = np.float32(np.mean(useg_rs))
    ustd = np.float32(np.std(useg_rs, ddof=1))
    slope = np.float32((useg_rs[-1] - useg_rs[0]) / useg_rs.size)
    diff_mid_vs_segmean = np.float32(mid - umean)
    uenergy = np.float32(np.trapezoid(useg_rs) / useg_rs.size)
    du = np.diff(useg_rs)
    max_slope = np.float32(np.max(du))
    min_slope = np.float32(np.min(du))
    vmax_diff_3p7 = np.float32(hi - 4.2)
    t_mean = np.float32(np.mean(tseg_rs))
    t_duration = np.float32(seg_duration)
    seg_capacity_s = np.float32(seg_capacity)
    seg_avg_power_s = np.float32(seg_avg_power)
    cond_vec = np.asarray(
        [
            chg_val,
            dsc_val,
            seg_avg_power_s,
            np.float32(protocol_id),
            np.float32(protocol_id * 2.5),
            np.float32(protocol_id * 3.8),
            vmin_n,
            vmax_n,
            mid,
            np.float32(vmax_n - mid),
            np.float32(protocol_id * 5),
            umean,
            ustd,
            slope,
            diff_mid_vs_segmean,
            uenergy,
            max_slope,
            min_slope,
            vmax_diff_3p7,
            t_mean,
            t_duration,
            seg_capacity_s,
        ],
        dtype=np.float32,
    )
    return cond_vec.reshape(1, -1)


def process_battery(config: DatasetConfig, battery_id: int, battery_full: list, battery_fix: list) -> dict:
    # Generate all MATLAB output variables for one battery.
    full_cycles = cycles_of(battery_full[battery_id - 1])
    fix_cycles = cycles_of(battery_fix[battery_id - 1])
    n_cycles = min(len(full_cycles), len(fix_cycles))
    gaf_values, frag_rows, seg_idx_rows, info_values, seg_ranges_values, cond_rows = [], [], [], [], [], []
    for cycle_index in range(1, n_cycles + 1):
        cycle = full_cycles[cycle_index - 1]
        fix_cycle = fix_cycles[cycle_index - 1]
        uraw = as_vector(cycle.voltage_V)
        traw = as_vector(cycle.time_s)
        iraw = as_vector(cycle.current_A)
        ugaf_rs, ugaf_interpolated = interp_to_size(as_vector(fix_cycle.voltage_V))
        tgaf_rs, _ = interp_to_size(as_vector(fix_cycle.time_s))
        t_01_full = (tgaf_rs - np.min(tgaf_rs)) / max(1e-9, np.max(tgaf_rs) - np.min(tgaf_rs))
        u01 = np.clip((ugaf_rs - U_MIN) / max(1e-9, U_MAX - U_MIN), 0, 1)
        full_gaf = matlab_gaf(t_01_full * np.pi, u01 * np.pi, ugaf_interpolated)
        gaf_values.append(to_uint8_image(full_gaf))
        info_values.append(np.asarray([[battery_id, cycle_index]], dtype=np.float64))
        starts = colon(float(np.min(uraw)), EFFECTIVE_STEP, float(np.max(uraw) - SEGMENT_SIZE))
        if starts.size == 0:
            starts = np.asarray([float(np.min(uraw))])
        if starts[-1] + SEGMENT_SIZE < float(np.max(uraw)) - 1e-9:
            starts = np.append(starts, float(np.max(uraw) - SEGMENT_SIZE))
        ranges = np.column_stack((starts, np.minimum(starts + SEGMENT_SIZE, float(np.max(uraw)))))
        frag_items, idx_items, cond_items, valid_ranges = [], [], [], []
        for lo, hi in ranges:
            mask = (uraw >= lo) & (uraw <= hi)
            useg = uraw[mask]
            tseg = traw[mask]
            iseg = iraw[mask]
            if useg.size == 0:
                useg = np.asarray([lo, hi], dtype=np.float64)
                tseg = np.asarray([traw[0], traw[-1]], dtype=np.float64)
                iseg = np.asarray([iraw[0], iraw[-1]], dtype=np.float64)
            elif useg.size == 1:
                useg = np.asarray([useg[0], useg[0] + max(1e-6, hi - lo)], dtype=np.float64)
                tseg = np.asarray([tseg[0], tseg[0] + 1e-3], dtype=np.float64)
                iseg = np.asarray([iseg[0], iseg[0]], dtype=np.float64)
            useg_rs, useg_interpolated = interp_to_size(useg)
            tseg_rs, _ = interp_to_size(tseg)
            t_seg_01 = (tseg_rs - np.min(tseg_rs)) / max(1e-9, np.max(tseg_rs) - np.min(tseg_rs))
            useg01 = np.clip((useg_rs - lo) / max(1e-9, hi - lo), 0, 1)
            frag_gaf = matlab_gaf(t_seg_01 * np.pi, useg01 * np.pi, useg_interpolated)
            previous_cycle = full_cycles[cycle_index - 2] if cycle_index > 1 else None
            cond_vec = make_cond_vec(cycle, previous_cycle, cycle_index, lo, hi, useg_rs, tseg_rs, iseg, tseg, useg, config.nca)
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
    frag_cell = empty_cell((rows, max_cols))
    seg_idx_cell = empty_cell((rows, max_cols))
    cond_vec_cell = empty_cell((rows, max_cols))
    for row in range(rows):
        gaf_cell[row, 0] = gaf_values[row]
        info_cell[row, 0] = info_values[row]
        seg_ranges_all[row, 0] = seg_ranges_values[row]
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
        "COND_VEC_cell": cond_vec_cell,
    }


def main() -> None:
    # Process every configured dataset and mirror all sliding MAT files into data/.
    output_dir = ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    for config in DATASETS:
        dataset_dir = ROOT / config.name
        battery_full, battery_fix = load_batteries(dataset_dir)
        nbat = min(19, len(battery_full), len(battery_fix))
        for battery_id in range(1, nbat + 1):
            output = process_battery(config, battery_id, battery_full, battery_fix)
            filename = dataset_split_name(battery_id, config)
            output_path = dataset_dir / filename
            savemat(output_path, output, do_compression=False)
            shutil.copy2(output_path, output_dir / filename)
            print(f"Saved -> {output_path.relative_to(ROOT)}")
    print("Done. All sliding MAT files were copied to data/.")


if __name__ == "__main__":
    main()
