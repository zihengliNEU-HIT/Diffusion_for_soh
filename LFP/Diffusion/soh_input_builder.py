from __future__ import annotations
import math
import os
import re
import warnings
import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.io import loadmat, savemat

REAL_DIR = "./data_for_generated"
GENERATED_DIR = "./generated_gaf/test_random"
OUTPUT_DIR = "./soh_input/test_random"
MATLAB_SOURCES = {
    "53C_54": "./matlab/matlab53C_54.mat",
    "56C_19": "./matlab/matlab56C_19.mat",
    "56C_36": "./matlab/matlab56C_36.mat",
}
STRICT_REQUIRE_REAL_FOR_TRAIN = True
STRICT_REQUIRE_GENERATED = True
COMPARE_SPLITS = {"train", "test"}
AUTO_SCALE_IMAGE_TO_01 = True
EXPECTED_TRAIN_WINDOWS = 3
VERBOSE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SOH input MAT files from generated GAF files."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON configuration file."
    )
    parser.add_argument("--real_dir", type=str, default=REAL_DIR)
    parser.add_argument("--generated_dir", type=str, default=GENERATED_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument(
        "--strict_require_real_for_train",
        action="store_true",
        default=STRICT_REQUIRE_REAL_FOR_TRAIN,
    )
    parser.add_argument(
        "--allow_missing_real_for_train", action="store_true", default=False
    )
    parser.add_argument(
        "--strict_require_generated",
        action="store_true",
        default=STRICT_REQUIRE_GENERATED,
    )
    parser.add_argument(
        "--auto_scale_image_to_01", action="store_true", default=AUTO_SCALE_IMAGE_TO_01
    )
    parser.add_argument(
        "--expected_train_windows", type=int, default=EXPECTED_TRAIN_WINDOWS
    )
    parser.add_argument("--verbose", action="store_true", default=VERBOSE)
    return parser.parse_args()


def apply_runtime_config(args: argparse.Namespace) -> None:
    config = {}
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    def value(name: str):
        return config.get(name, getattr(args, name))

    global REAL_DIR, GENERATED_DIR, OUTPUT_DIR
    global STRICT_REQUIRE_REAL_FOR_TRAIN, STRICT_REQUIRE_GENERATED
    global AUTO_SCALE_IMAGE_TO_01, EXPECTED_TRAIN_WINDOWS, VERBOSE
    global MATLAB_SOURCES, COMPARE_SPLITS
    REAL_DIR = value("real_dir")
    GENERATED_DIR = value("generated_dir")
    OUTPUT_DIR = value("output_dir")
    STRICT_REQUIRE_REAL_FOR_TRAIN = bool(value("strict_require_real_for_train"))
    if bool(value("allow_missing_real_for_train")):
        STRICT_REQUIRE_REAL_FOR_TRAIN = False
    STRICT_REQUIRE_GENERATED = bool(value("strict_require_generated"))
    AUTO_SCALE_IMAGE_TO_01 = bool(value("auto_scale_image_to_01"))
    EXPECTED_TRAIN_WINDOWS = int(value("expected_train_windows"))
    VERBOSE = bool(value("verbose"))
    if "matlab_sources" in config:
        MATLAB_SOURCES = dict(config["matlab_sources"])
    if "compare_splits" in config:
        COMPARE_SPLITS = set(config["compare_splits"])


REAL_RE = re.compile(
    "(?P<chem>53C_54|56C_19|56C_36)LFP_battery(?P<batt>\\d+).*?_(?P<split>train|val|test)_sliding\\.mat$",
    re.IGNORECASE,
)
GEN_RE = re.compile(
    "(?P<chem>53C_54|56C_19|56C_36)LFP_battery(?P<batt>\\d+).*?_(?P<split>train|val|test)_20gaf\\.mat$",
    re.IGNORECASE,
)
BATTERY_HINT_RE = re.compile("battery(?P<batt>\\d+)", re.IGNORECASE)


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_chem(chem: str) -> str:
    chem = chem.upper()
    if chem in {"53C_54", "56C_19", "56C_36"}:
        return chem
    raise ValueError(f"Unknown chemistry: {chem}")


def parse_key(fname: str, kind: str) -> Optional[Tuple[str, int, str]]:
    regex = REAL_RE if kind == "real" else GEN_RE
    m = regex.search(fname)
    if not m:
        return None
    chem = normalize_chem(m.group("chem"))
    batt = int(m.group("batt"))
    split = m.group("split").lower()
    return (chem, batt, split)


def key_to_stem(key: Tuple[str, int, str]) -> str:
    chem, batt, split = key
    return f"{chem}LFP_battery{batt}_{split}"


def maybe_scale_img01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if AUTO_SCALE_IMAGE_TO_01:
        mx = np.nanmax(arr) if arr.size > 0 else 0.0
        if mx > 1.0:
            arr = arr / 255.0
    return arr.astype(np.float32)


def to_float_image(x: Any) -> np.ndarray:
    img = np.asarray(x, dtype=np.float32)
    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape={img.shape}")
    return maybe_scale_img01(img)


def to_info_matrix(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"Invalid INFO 1D shape: {arr.shape}")
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Invalid INFO dimensions: {arr.shape}")
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"INFO must be [N,2], got {arr.shape}")
    return arr.astype(np.int32)


def safe_listify(obj: Any) -> List[Any]:
    if isinstance(obj, np.ndarray):
        return list(obj.ravel())
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def safe_get(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


def squeeze_scalar(x: Any) -> Any:
    arr = np.asarray(x)
    if arr.size == 1:
        return arr.reshape(-1)[0]
    return x


def scan_folder(folder: str, kind: str) -> Dict[Tuple[str, int, str], str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    out: Dict[Tuple[str, int, str], str] = {}
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath) or not fname.lower().endswith(".mat"):
            continue
        key = parse_key(fname, kind)
        if key is not None:
            out[key] = fpath
    return out


def load_generated_file(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    mat = loadmat(path)
    if "GAF_gen" not in mat or "INFO" not in mat:
        raise KeyError(f"{os.path.basename(path)} is missing GAF_gen or INFO fields")
    gaf_gen = np.asarray(mat["GAF_gen"], dtype=np.float32)
    info = to_info_matrix(mat["INFO"])
    seg_idx = None
    if "SEGMENT_IDX" in mat:
        seg_idx = np.asarray(mat["SEGMENT_IDX"]).reshape(-1)
        seg_idx = seg_idx.astype(np.int32)
    if gaf_gen.ndim != 4:
        raise ValueError(f"GAF_gen must be [N,K,H,W], got {gaf_gen.shape}")
    if gaf_gen.shape[0] != info.shape[0]:
        raise ValueError(
            f"N mismatch: GAF_gen={gaf_gen.shape[0]}, INFO={info.shape[0]}"
        )
    if seg_idx is not None and seg_idx.shape[0] != gaf_gen.shape[0]:
        raise ValueError(
            f"N mismatch: GAF_gen={gaf_gen.shape[0]}, SEGMENT_IDX={seg_idx.shape[0]}"
        )
    if AUTO_SCALE_IMAGE_TO_01 and np.nanmax(gaf_gen) > 1.0:
        gaf_gen = gaf_gen / 255.0
    return (gaf_gen.astype(np.float32), info.astype(np.int32), seg_idx)


def parse_info_entry(entry: Any, default_cycle_idx: int) -> Tuple[int, int]:
    arr = np.asarray(entry)
    arr = np.squeeze(arr)
    if arr.size >= 2:
        flat = arr.reshape(-1)
        return (int(flat[0]), int(flat[1]))
    return (0, int(default_cycle_idx))


def build_real_gaf_map(real_mat: Dict[str, Any]) -> Dict[int, np.ndarray]:
    if "GAF_cell" not in real_mat:
        raise KeyError("real sliding MAT is missing GAF_cell")
    gaf_cell = np.asarray(real_mat["GAF_cell"], dtype=object)
    info_cell = real_mat.get("info_cell", None)
    n_cycles = gaf_cell.shape[0]
    out: Dict[int, np.ndarray] = {}
    for i in range(n_cycles):
        img = gaf_cell[i, 0] if gaf_cell.ndim >= 2 else gaf_cell[i]
        img_arr = np.asarray(img)
        if img_arr.size == 0:
            continue
        img2d = to_float_image(img_arr)
        if info_cell is not None:
            entry = info_cell[i, 0] if np.asarray(info_cell).ndim >= 2 else info_cell[i]
            _, cycle_idx = parse_info_entry(entry, i + 1)
        else:
            cycle_idx = i + 1
        out[int(cycle_idx)] = img2d
    return out


def align_real_by_info(real_map: Dict[int, np.ndarray], info: np.ndarray) -> np.ndarray:
    aligned: List[np.ndarray] = []
    missing: List[int] = []
    for _, cycle_idx in info:
        c = int(cycle_idx)
        if c not in real_map:
            missing.append(c)
        else:
            aligned.append(real_map[c])
    if missing:
        raise KeyError(
            f"real GAF is missing cycle_idx: {missing[:10]}{('...' if len(missing) > 10 else '')}"
        )
    return np.stack(aligned, axis=0).astype(np.float32)


def extract_soh_list_from_single_mat(
    mat_data: Dict[str, Any], batt_id: Optional[int]
) -> Optional[List[float]]:
    if "SOH_cell" in mat_data:
        cell = np.asarray(mat_data["SOH_cell"], dtype=object).ravel()
        vals: List[float] = []
        for x in cell:
            try:
                vals.append(float(np.asarray(x).reshape(-1)[0]))
            except Exception:
                vals.append(np.nan)
        return vals
    for key in ["battery", "battery_full"]:
        if key in mat_data:
            if batt_id is None:
                raise ValueError(f"{key} requires batt_id in struct mode")
            batt_list = safe_listify(mat_data[key])
            if batt_id < 1 or batt_id > len(batt_list):
                raise IndexError(
                    f"battery_id={batt_id} is out of range, total={len(batt_list)}"
                )
            batt_obj = batt_list[batt_id - 1]
            cycles = safe_get(batt_obj, "cycles", None)
            if cycles is None:
                raise KeyError(f"{key}  battery({batt_id}) has no cycles field")
            vals = []
            for cyc in safe_listify(cycles):
                if safe_get(cyc, "capacity", None) is not None:
                    cap = safe_get(cyc, "capacity", np.nan)
                elif safe_get(cyc, "SOH", None) is not None:
                    cap = safe_get(cyc, "SOH", np.nan)
                else:
                    cap = np.nan
                try:
                    vals.append(float(squeeze_scalar(cap)))
                except Exception:
                    vals.append(np.nan)
            return vals
    return None


def find_battery_file_in_dir(folder: str, batt_id: int) -> str:
    candidates = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath) or not fname.lower().endswith(".mat"):
            continue
        m = BATTERY_HINT_RE.search(fname)
        if m and int(m.group("batt")) == batt_id:
            candidates.append(fpath)
    if not candidates:
        raise FileNotFoundError(
            f"No matching {folder} found in directory battery{batt_id} .mat file"
        )
    return candidates[0]


def load_capacity_list(source_path: str, batt_id: int) -> List[float]:
    if os.path.isdir(source_path):
        mat_path = find_battery_file_in_dir(source_path, batt_id)
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        vals = extract_soh_list_from_single_mat(data, batt_id=None)
        if vals is None:
            vals = extract_soh_list_from_single_mat(data, batt_id=batt_id)
        if vals is None:
            raise KeyError(f"{mat_path} has no usable SOH_cell or battery structure")
        return vals
    if os.path.isfile(source_path):
        data = loadmat(source_path, squeeze_me=True, struct_as_record=False)
        vals = extract_soh_list_from_single_mat(data, batt_id=batt_id)
        if vals is None:
            raise KeyError(f"{source_path} has no usable SOH_cell or battery structure")
        return vals
    raise FileNotFoundError(f"matlab source does not exist: {source_path}")


def align_capacity_by_info(cap_list: Sequence[float], info: np.ndarray) -> np.ndarray:
    out: List[float] = []
    for _, cycle_idx in info:
        c = int(cycle_idx)
        if 1 <= c <= len(cap_list):
            out.append(float(cap_list[c - 1]))
        else:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float32).reshape(-1, 1)


def aggregate_train_windows(
    gaf_gen: np.ndarray,
    info: np.ndarray,
    seg_idx: Optional[np.ndarray] = None,
    expected_windows: int = EXPECTED_TRAIN_WINDOWS,
) -> Tuple[np.ndarray, np.ndarray]:
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, row in enumerate(info):
        key = (int(row[0]), int(row[1]))
        groups[key].append(i)
    gaf_out: List[np.ndarray] = []
    info_out: List[List[int]] = []
    for key in sorted(groups.keys()):
        idxs = groups[key]
        if seg_idx is not None:
            idxs = sorted(idxs, key=lambda i: (int(seg_idx[i]), i))
        else:
            idxs = sorted(idxs)
        if len(idxs) != expected_windows:
            raise ValueError(
                f"train cycle {key} has {len(idxs)} windows, expected {expected_windows}. Check that reverse diffusion train output keeps 3 windows per cycle."
            )
        parts = [gaf_gen[i] for i in idxs]
        merged = np.concatenate(parts, axis=0)
        gaf_out.append(merged.astype(np.float32))
        info_out.append([key[0], key[1]])
    return (np.stack(gaf_out, axis=0), np.asarray(info_out, dtype=np.int32))


def calculate_metrics(target: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    mse = float(np.mean((target - generated) ** 2))
    if mse == 0:
        psnr = 100.0
    else:
        max_pixel = 1.0
        psnr = 20.0 * math.log10(max_pixel) - 10.0 * math.log10(mse)
    target_mean = float(np.mean(target))
    generated_mean = float(np.mean(generated))
    target_var = float(np.var(target))
    generated_var = float(np.var(generated))
    target_generated_cov = float(
        np.mean((target - target_mean) * (generated - generated_mean))
    )
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    luminance = (2 * target_mean * generated_mean + C1) / (
        target_mean**2 + generated_mean**2 + C1
    )
    contrast = (
        2 * math.sqrt(max(target_var, 0.0)) * math.sqrt(max(generated_var, 0.0)) + C2
    ) / (target_var + generated_var + C2)
    structure = (target_generated_cov + C2 / 2) / (
        math.sqrt(max(target_var, 0.0)) * math.sqrt(max(generated_var, 0.0)) + C2 / 2
    )
    ssim = float(luminance * contrast * structure)
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def compute_cycle_metrics(
    gaf_real: np.ndarray, gaf_gen: np.ndarray
) -> List[Dict[str, Any]]:
    N, K = gaf_gen.shape[:2]
    rows: List[Dict[str, Any]] = []
    for i in range(N):
        real = gaf_real[i]
        gen_stack = gaf_gen[i]
        gen_mean = np.mean(gen_stack, axis=0).astype(np.float32)
        mean_metrics = calculate_metrics(real, gen_mean)
        diff = gen_mean - real
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        psnr_samples: List[float] = []
        for k in range(K):
            mk = calculate_metrics(real, gen_stack[k])
            psnr_samples.append(float(mk["psnr"]))
        psnr_samples_arr = np.asarray(psnr_samples, dtype=np.float32)
        rows.append(
            {
                "mean_mse": float(mean_metrics["mse"]),
                "mean_psnr": float(mean_metrics["psnr"]),
                "mean_ssim": float(mean_metrics["ssim"]),
                "mae": mae,
                "rmse": rmse,
                "psnr_samples": psnr_samples,
                "psnr_sample_mean": float(np.mean(psnr_samples_arr)),
                "psnr_sample_std": float(np.std(psnr_samples_arr)),
                "psnr_sample_min": float(np.min(psnr_samples_arr)),
                "psnr_sample_max": float(np.max(psnr_samples_arr)),
                "psnr_sample_median": float(np.median(psnr_samples_arr)),
            }
        )
    return rows


def write_compare_txt(
    txt_path: str,
    key: Tuple[str, int, str],
    source_real: str,
    source_gen: str,
    source_soh: str,
    info: np.ndarray,
    capacity: np.ndarray,
    cycle_metrics: List[Dict[str, Any]],
) -> None:
    chem, batt, split = key
    cycle_ids = info[:, 1].astype(int).tolist()
    caps = capacity.reshape(-1).tolist()
    mean_psnr_all = (
        np.mean([r["mean_psnr"] for r in cycle_metrics])
        if cycle_metrics
        else float("nan")
    )
    mean_ssim_all = (
        np.mean([r["mean_ssim"] for r in cycle_metrics])
        if cycle_metrics
        else float("nan")
    )
    mean_mae_all = (
        np.mean([r["mae"] for r in cycle_metrics]) if cycle_metrics else float("nan")
    )
    mean_rmse_all = (
        np.mean([r["rmse"] for r in cycle_metrics]) if cycle_metrics else float("nan")
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SOH comparison report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Key                : {chem}battery{batt}_{split}\n")
        f.write(f"Real source        : {source_real}\n")
        f.write(f"Generated source   : {source_gen}\n")
        f.write(f"SOH source         : {source_soh}\n")
        f.write(f"Number of cycles   : {len(cycle_metrics)}\n\n")
        f.write("Overall summary\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean-image PSNR avg : {mean_psnr_all:.6f} dB\n")
        f.write(f"Mean-image SSIM avg : {mean_ssim_all:.6f}\n")
        f.write(f"Diff MAE avg        : {mean_mae_all:.6f}\n")
        f.write(f"Diff RMSE avg       : {mean_rmse_all:.6f}\n\n")
        f.write("Per-cycle metrics\n")
        f.write("-" * 80 + "\n")
        f.write(
            "cycle_idx\tcapacity\tmae\trmse\tmean_mse\tmean_psnr\tmean_ssim\tpsnr_sample_mean\tpsnr_sample_std\tpsnr_sample_min\tpsnr_sample_max\tpsnr_sample_median\n"
        )
        for cycle_idx, cap, row in zip(cycle_ids, caps, cycle_metrics):
            f.write(
                f"{cycle_idx}\t{cap:.10f}\t{row['mae']:.10f}\t{row['rmse']:.10f}\t{row['mean_mse']:.10f}\t{row['mean_psnr']:.10f}\t{row['mean_ssim']:.10f}\t{row['psnr_sample_mean']:.10f}\t{row['psnr_sample_std']:.10f}\t{row['psnr_sample_min']:.10f}\t{row['psnr_sample_max']:.10f}\t{row['psnr_sample_median']:.10f}\n"
            )


def write_soh_mat(
    out_path: str,
    split: str,
    gaf_gen: np.ndarray,
    capacity: np.ndarray,
    info: np.ndarray,
    gaf_real: Optional[np.ndarray] = None,
) -> None:
    out = {
        "GAF_gen": gaf_gen.astype(np.float32),
        "CAPACITY": capacity.astype(np.float32),
        "INFO": info.astype(np.int32),
    }
    if split == "train":
        if gaf_real is None:
            raise ValueError("train output must include GAF_real")
        out["GAF_real"] = gaf_real.astype(np.float32)
    savemat(out_path, out)


def main() -> None:
    apply_runtime_config(parse_args())
    ensure_dir(OUTPUT_DIR)
    compare_dir = os.path.join(OUTPUT_DIR, "comparisons")
    ensure_dir(compare_dir)
    real_files = scan_folder(REAL_DIR, kind="real")
    gen_files = scan_folder(GENERATED_DIR, kind="generated")
    log(f"[scan] real files      : {len(real_files)}")
    log(f"[scan] generated files : {len(gen_files)}")
    if STRICT_REQUIRE_GENERATED and (not gen_files):
        raise RuntimeError("No *_20gaf.mat files found in generated_dir")
    summary_lines: List[str] = []
    saved_count = 0
    skipped_count = 0
    for key in sorted(gen_files.keys(), key=lambda x: (x[2], x[0], x[1])):
        chem, batt, split = key
        gen_path = gen_files[key]
        real_path = real_files.get(key, None)
        matlab_source = MATLAB_SOURCES.get(chem, None)
        log(f"\n[build] {key_to_stem(key)}")
        log(f"  generated: {os.path.basename(gen_path)}")
        log(
            f"  real     : {(os.path.basename(real_path) if real_path else '<missing>')}"
        )
        log(f"  soh src  : {matlab_source}")
        if matlab_source is None:
            skipped_count += 1
            print(
                f"  [ERROR] {key_to_stem(key)} -> Missing MATLAB source config for {chem}"
            )
            continue
        if split == "train" and STRICT_REQUIRE_REAL_FOR_TRAIN and (real_path is None):
            skipped_count += 1
            print(f"  [ERROR] {key_to_stem(key)} -> missing real sliding file")
            continue
        try:
            gaf_gen_raw, info_raw, seg_idx = load_generated_file(gen_path)
            cap_list = load_capacity_list(matlab_source, batt_id=batt)
            if split == "train":
                gaf_gen, info = aggregate_train_windows(gaf_gen_raw, info_raw, seg_idx)
            else:
                gaf_gen, info = (gaf_gen_raw, info_raw)
            capacity = align_capacity_by_info(cap_list, info)
            gaf_real: Optional[np.ndarray] = None
            if split == "train":
                if real_path is None:
                    raise FileNotFoundError(f"{key_to_stem(key)} missing real file")
                real_mat = loadmat(real_path)
                real_map = build_real_gaf_map(real_mat)
                gaf_real = align_real_by_info(real_map, info)
            out_name = f"{chem}battery{batt}_{split}_soh.mat"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            write_soh_mat(
                out_path=out_path,
                split=split,
                gaf_gen=gaf_gen,
                capacity=capacity,
                info=info,
                gaf_real=gaf_real,
            )
            saved_count += 1
            log(f"  saved    : {out_name}")
            log(f"  GAF_gen  : {gaf_gen.shape}")
            if gaf_real is not None:
                log(f"  GAF_real : {gaf_real.shape}")
            nan_cap = int(np.isnan(capacity).sum())
            if nan_cap > 0:
                warnings.warn(
                    f"{out_name} has {nan_cap} CAPACITY values as NaN; check the corresponding MATLAB source"
                )
            if split in COMPARE_SPLITS and gaf_real is not None:
                cycle_metrics = compute_cycle_metrics(gaf_real, gaf_gen)
                txt_name = f"{chem}battery{batt}_{split}_compare.txt"
                txt_path = os.path.join(compare_dir, txt_name)
                write_compare_txt(
                    txt_path=txt_path,
                    key=key,
                    source_real=os.path.basename(real_path),
                    source_gen=os.path.basename(gen_path),
                    source_soh=os.path.basename(str(matlab_source)),
                    info=info,
                    capacity=capacity,
                    cycle_metrics=cycle_metrics,
                )
                log(f"  compare  : {txt_name}")
                summary_lines.append(
                    f"{chem}battery{batt}_{split}\tn_cycles={len(cycle_metrics)}\tmean_psnr={np.mean([r['mean_psnr'] for r in cycle_metrics]):.6f}\tmean_ssim={np.mean([r['mean_ssim'] for r in cycle_metrics]):.6f}\tmean_mae={np.mean([r['mae'] for r in cycle_metrics]):.6f}\tmean_rmse={np.mean([r['rmse'] for r in cycle_metrics]):.6f}"
                )
        except Exception as e:
            skipped_count += 1
            print(f"  [ERROR] {key_to_stem(key)} -> {e}")
    summary_path = os.path.join(compare_dir, "comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Comparison summary\n")
        f.write("=" * 80 + "\n")
        if summary_lines:
            for line in summary_lines:
                f.write(line + "\n")
        else:
            f.write("No comparison report generated.\n")
    log(f"\n[summary] {summary_path}")
    print("\n================ DONE ================")
    print(f"Saved SOH mats : {saved_count}")
    print(f"Skipped items  : {skipped_count}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print(f"Compare dir    : {compare_dir}")


if __name__ == "__main__":
    main()
