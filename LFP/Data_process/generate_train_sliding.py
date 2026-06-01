#!/usr/bin/env python3
"""Generate LFP sliding-window MAT files from the original MATLAB workflow."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parent
IMG_SIZE = 64
SEGMENT_SIZE = 0.2
OVERLAP = 0.4
EFFECTIVE_STEP = SEGMENT_SIZE * (1 - OVERLAP)
V_MIN = 3.0
V_MAX = 4.2


@dataclass(frozen=True)
class DatasetConfig:
    # Static settings copied from each LFP code3_slidingwindows.m script.
    name: str
    mat_file: str
    prefix: str
    protocol_id: int
    train_set: tuple[int, ...]
    val_set: tuple[int, ...]
    test_set: tuple[int, ...]


DATASETS = (
    # 5_6C_19 split: train=[1,2,4,5], val=[3], test=[6,7].
    DatasetConfig("5_6C_19", "5_6C_19PER_4_6C_NEWSTRUCTURE_resampled.mat", "56C_19LFP", 3, (1, 2, 4, 5), (3,), (6, 7)),
    # 5_6C_36 split: train=[2,4,5,7], val=[8,3], test=[1,6].
    DatasetConfig("5_6C_36", "5_6C_36PER_4_3C_NEWSTRUCTURE_resampled.mat", "56C_36LFP", 2, (2, 4, 5, 7), (8, 3), (1, 6)),
    # 5_3C_54 split: train=[1,6,3,5,7], val=[4,2], test=[8].
    DatasetConfig("5_3C_54", "5_3C_54PER_4C_NEWSTRUCTURE_resampled.mat", "53C_54LFP", 1, (1, 6, 3, 5, 7), (4, 2), (8,)),
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


def get_field(struct_item, field_name: str):
    # Read a field from scipy's MATLAB struct representation.
    value = getattr(struct_item, field_name)
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        return value.flat[0]
    return value


def load_batteries(dataset_dir: Path, mat_file: str) -> list:
    # Load the single resampled MATLAB file used by the LFP scripts.
    data = loadmat(dataset_dir / mat_file, squeeze_me=False, struct_as_record=False)
    return matlab_struct_array(data["battery"])


def interp_to_size(values: np.ndarray, size: int = IMG_SIZE) -> tuple[np.ndarray, bool]:
    # Reproduce MATLAB interp1(..., 'linear') when the input length is not IMG_SIZE.
    values = as_vector(values)
    if values.size == size:
        return values, False
    x_old = np.arange(1, values.size + 1, dtype=np.float64)
    x_new = np.linspace(1, values.size, size)
    return np.interp(x_new, x_old, values), True


def matlab_gaf(theta_t: np.ndarray, theta_v: np.ndarray, interpolated: bool) -> np.ndarray:
    # Preserve MATLAB row/column orientation after interp1 for GAF construction.
    if interpolated:
        return np.cos(theta_t[None, :] + theta_v[:, None])
    return np.cos(theta_t[:, None] + theta_v[None, :])


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
    cell.fill(np.empty((1, 0), dtype=np.float64))
    return cell


def split_name_for_battery(battery_id: int, config: DatasetConfig) -> str:
    # Use the same split and filename rules as the MATLAB scripts.
    if battery_id in config.train_set:
        return "train"
    if battery_id in config.val_set:
        return "val"
    if battery_id in config.test_set:
        return "test"
    return "other"


def build_cond_vec(config: DatasetConfig, cycle_index: int, lo: float, hi: float, v_seg_rs: np.ndarray,
                   t_seg_rs: np.ndarray, v_seg: np.ndarray, i_seg: np.ndarray, seg_capacity: float,
                   seg_avg_power: float, seg_power_density: float, seg_capacity_ln: float,
                   seg_duration: float) -> np.ndarray:
    # Build the condition vector in the original MATLAB order.
    vmin_n = lo
    vmax_n = hi
    width = vmax_n - vmin_n
    mid = (vmin_n + vmax_n) / 2
    cycle_num = np.float32(cycle_index)
    cycle_sqrt = np.float32(np.sqrt(cycle_index))
    cycle_log = np.float32(np.log(1 + cycle_index))
    vmax_diff_3p7 = np.float32(hi - 4.2)
    protocol_id = np.float32(config.protocol_id)
    v_mean = float(np.mean(v_seg_rs))
    v_std = float(np.std(v_seg_rs, ddof=1))
    slope = float((v_seg_rs[-1] - v_seg_rs[0]) / v_seg_rs.size)
    diff_max_mid = np.float32(V_MAX - mid)
    v_energy = float(np.trapezoid(v_seg_rs) / v_seg_rs.size)
    dv = np.diff(v_seg_rs)
    max_slope = float(np.max(dv))
    min_slope = float(np.min(dv))
    t_mean = float(np.mean(t_seg_rs))
    t_duration = float(np.max(t_seg_rs) - np.min(t_seg_rs))
    del v_seg, i_seg
    cond_vec = np.asarray(
        [
            vmin_n,
            vmax_n,
            width,
            mid,
            cycle_num,
            cycle_sqrt,
            cycle_log,
            vmax_diff_3p7,
            protocol_id,
            np.float32(protocol_id * 2.5),
            v_mean,
            v_std,
            slope,
            diff_max_mid,
            v_energy,
            max_slope,
            min_slope,
            t_mean,
            t_duration,
            seg_capacity,
            seg_avg_power,
            seg_power_density,
            seg_capacity_ln,
            seg_duration,
        ],
        dtype=np.float32,
    )
    return cond_vec.reshape(1, -1)


def process_battery(config: DatasetConfig, battery_id: int, battery_item) -> dict:
    # Generate all MATLAB output variables for one battery.
    cycles = cycles_of(battery_item)
    gaf_values, frag_rows, seg_idx_rows, info_values, seg_ranges_values, cond_rows = [], [], [], [], [], []
    for cycle_index, cycle in enumerate(cycles, start=1):
        phase2 = get_field(cycle, "phase2")
        t_phase2 = as_vector(get_field(phase2, "t"))
        v_phase2 = as_vector(get_field(phase2, "V"))
        if t_phase2.size != IMG_SIZE or v_phase2.size != IMG_SIZE:
            continue
        t_01 = (t_phase2 - np.min(t_phase2)) / max(1e-9, np.max(t_phase2) - np.min(t_phase2))
        v_01 = np.clip((v_phase2 - V_MIN) / max(1e-9, V_MAX - V_MIN), 0, 1)
        full_gaf = np.cos((t_01 * np.pi)[:, None] + (v_01 * np.pi)[None, :])
        gaf_values.append(to_uint8_image(full_gaf))
        row = len(gaf_values)
        info_values.append(np.asarray([[battery_id, cycle_index]], dtype=np.float64))
        phase1 = get_field(cycle, "phase1")
        t_phase1 = as_vector(get_field(phase1, "t"))
        v_phase1 = as_vector(get_field(phase1, "V"))
        i_phase1 = as_vector(get_field(phase1, "I"))
        v_min_cycle = float(np.min(v_phase1))
        v_max_cycle = float(np.max(v_phase1))
        starts = colon(v_min_cycle, EFFECTIVE_STEP, v_max_cycle - SEGMENT_SIZE)
        if starts.size == 0:
            starts = np.asarray([v_min_cycle], dtype=np.float64)
        if starts[-1] + SEGMENT_SIZE < v_max_cycle - 1e-9:
            starts = np.append(starts, v_max_cycle - SEGMENT_SIZE)
        ranges = np.column_stack((starts, np.minimum(starts + SEGMENT_SIZE, v_max_cycle)))
        frag_items, idx_items, cond_items, valid_ranges = [], [], [], []
        for lo, hi in ranges:
            mask = (v_phase1 >= lo) & (v_phase1 <= hi)
            t_seg = t_phase1[mask]
            v_seg = v_phase1[mask]
            i_seg = i_phase1[mask]
            if v_seg.size == 0:
                v_seg = np.asarray([lo, hi], dtype=np.float64)
                t_seg = np.asarray([t_phase1[0], t_phase1[-1]], dtype=np.float64)
                i_seg = np.asarray([i_phase1[0], i_phase1[-1]], dtype=np.float64)
            elif v_seg.size == 1:
                v_seg = np.asarray([v_seg[0], v_seg[0] + max(1e-6, hi - lo)], dtype=np.float64)
                t_seg = np.asarray([t_seg[0], t_seg[0] + 1e-3], dtype=np.float64)
                i_seg = np.asarray([i_seg[0], i_seg[0]], dtype=np.float64)
            if t_seg.size > 1:
                dt_hours = np.diff(t_seg) / 3600
                i_avg = (i_seg[:-1] + i_seg[1:]) / 2
                seg_capacity = float(np.sum(np.abs(i_avg * dt_hours)))
            else:
                seg_capacity = 0.0
            seg_avg_power = float(np.mean(np.abs(v_seg * i_seg)))
            seg_power_density = seg_avg_power / seg_capacity if seg_capacity > 1e-9 else 0.0
            seg_capacity_ln = float(np.log(1 + seg_capacity))
            seg_duration = float(np.max(t_seg) - np.min(t_seg))
            t_seg_rs, t_interpolated = interp_to_size(t_seg)
            v_seg_rs, _ = interp_to_size(v_seg)
            t_seg_01 = (t_seg_rs - np.min(t_seg_rs)) / max(1e-9, np.max(t_seg_rs) - np.min(t_seg_rs))
            v_seg_01 = np.clip((v_seg_rs - lo) / max(1e-9, hi - lo), 0, 1)
            frag_gaf = matlab_gaf(t_seg_01 * np.pi, v_seg_01 * np.pi, t_interpolated)
            cond_vec = build_cond_vec(
                config,
                cycle_index,
                lo,
                hi,
                v_seg_rs,
                t_seg_rs,
                v_seg,
                i_seg,
                seg_capacity,
                seg_avg_power,
                seg_power_density,
                seg_capacity_ln,
                seg_duration,
            )
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
    # Write each dataset output in place, then stage train/val and test separately.
    data_dir = ROOT / "data"
    data_for_generated_dir = ROOT / "data_for_generated"
    data_dir.mkdir(exist_ok=True)
    data_for_generated_dir.mkdir(exist_ok=True)
    for config in DATASETS:
        dataset_dir = ROOT / config.name
        batteries = load_batteries(dataset_dir, config.mat_file)
        for battery_id, battery_item in enumerate(batteries, start=1):
            output = process_battery(config, battery_id, battery_item)
            split_name = split_name_for_battery(battery_id, config)
            filename = f"{config.prefix}_battery{battery_id}_{split_name}_sliding.mat"
            output_path = dataset_dir / filename
            savemat(output_path, output, do_compression=False)
            if split_name == "test":
                shutil.copy2(output_path, data_for_generated_dir / filename)
            elif split_name in {"train", "val"}:
                shutil.copy2(output_path, data_dir / filename)
            print(f"Saved -> {output_path.relative_to(ROOT)}")
    print("Done. Train/val files were copied to data/, and test files were copied to data_for_generated/.")


if __name__ == "__main__":
    main()
