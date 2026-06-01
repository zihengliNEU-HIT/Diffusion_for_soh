import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import re
import matplotlib

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


class BatteryCAFDataset(Dataset):

    def __init__(
        self,
        data_dir,
        split="train",
        transform=None,
        cond_mean=None,
        cond_std=None,
        val_window_mode="random",
        test_window_mode="middle",
        random_seed=42,
    ):
        self.data_dir = data_dir
        self.split = split.lower()
        self.transform = transform
        self.val_window_mode = str(val_window_mode).lower()
        self.test_window_mode = str(test_window_mode).lower()
        self.random_seed = int(random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        self.data_files = []
        if self.split == "train":
            pat = re.compile(".*_train_sliding\\.mat$", re.I)
        elif self.split == "val":
            pat = re.compile(".*_val_sliding\\.mat$", re.I)
        elif self.split == "test":
            pat = re.compile(".*_test_sliding\\.mat$", re.I)
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        for fname in os.listdir(data_dir):
            if pat.match(fname):
                self.data_files.append(os.path.join(data_dir, fname))
        if not self.data_files:
            print(
                f"Warning: no {data_dir} found for '*_{self.split}_sliding.mat' files"
            )
        self.total_cycles = 0
        self.skipped_empty_segments = 0
        self.valid_segments = 0
        self.selected_segments = 0
        self.flat_samples = []
        self.all_cond_vecs_raw = []
        for file in self.data_files:
            if os.path.exists(file):
                self._load_and_flatten_data(file)
            else:
                print(f"Warning: {file} does not exist")
        print(
            f"Loaded {split} with {len(self.flat_samples)} final samples (LMH window selection)"
        )
        print(
            f"Stats: total cycles {self.total_cycles}, valid windows {self.valid_segments}, selected windows {self.selected_segments}, skipped empty segments {self.skipped_empty_segments}"
        )
        if len(self.flat_samples) > 0:
            if cond_mean is not None and cond_std is not None:
                self.cond_mean = np.asarray(cond_mean, dtype=np.float32)
                self.cond_std = np.asarray(cond_std, dtype=np.float32)
                self.cond_std = np.where(self.cond_std < 1e-08, 1.0, self.cond_std)
                print(
                    f"Condition-vector z-score stats [{self.split}]: using train mean/std"
                )
            else:
                self._compute_cond_stats()
        else:
            self.cond_mean = np.zeros((1,), dtype=np.float32)
            self.cond_std = np.ones((1,), dtype=np.float32)

    def _compute_cond_stats(self):
        if len(self.all_cond_vecs_raw) == 0:
            self.cond_mean = np.zeros((1,), dtype=np.float32)
            self.cond_std = np.ones((1,), dtype=np.float32)
            return
        all_vecs = np.stack(self.all_cond_vecs_raw, axis=0)
        self.cond_mean = np.mean(all_vecs, axis=0)
        self.cond_std = np.std(all_vecs, axis=0)
        self.cond_std = np.where(self.cond_std < 1e-08, 1.0, self.cond_std)
        print(
            f"Condition-vector z-score stats [{self.split}]: based on all {len(self.all_cond_vecs_raw)} valid windows (before LMH selection)"
        )
        print(f"  mean={self.cond_mean}")
        print(f"  std ={self.cond_std}")

    def _to_1d_float_vec(self, x):
        arr = np.array(x, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = np.array([arr], dtype=np.float32)
        return arr.astype(np.float32)

    def _parse_cycle_info(self, cycle_info, default_cycle_idx):
        if isinstance(cycle_info, np.ndarray):
            if cycle_info.ndim == 2 and cycle_info.shape == (1, 2):
                battery_id = int(cycle_info[0, 0])
                cycle_idx = int(cycle_info[0, 1])
            elif cycle_info.ndim == 1 and cycle_info.size >= 2:
                battery_id = int(cycle_info[0])
                cycle_idx = int(cycle_info[1])
            else:
                flat_info = cycle_info.flatten()
                battery_id = int(flat_info[0])
                cycle_idx = int(flat_info[1])
            return [battery_id, cycle_idx]
        if isinstance(cycle_info, (list, tuple)):
            return [int(cycle_info[0]), int(cycle_info[1])]
        print(f"Warning: Unknown info format: {type(cycle_info)}")
        return [0, default_cycle_idx]

    def _resolve_segment_idx(self, seg_idx_cell, i, s):
        if seg_idx_cell is not None and seg_idx_cell[i, s].size > 0:
            return int(np.asarray(seg_idx_cell[i, s]).reshape(-1)[0])
        return int(s)

    def _is_valid_window(self, frag, cond):
        if frag is None or not isinstance(frag, np.ndarray) or frag.size == 0:
            return False
        if cond is None:
            return False
        if isinstance(cond, np.ndarray) and cond.size == 0:
            return False
        return True

    def _single_window_position(self, n_valid_windows, mode="middle"):
        if n_valid_windows <= 0:
            return None
        mode = str(mode).lower()
        low_pos = min(3, n_valid_windows - 1)
        high_pos = max(n_valid_windows - 3, 0)
        if high_pos < low_pos:
            high_pos = low_pos
        middle_pos = (low_pos + high_pos) // 2
        if mode == "low":
            return low_pos
        if mode == "high":
            return high_pos
        if mode == "middle":
            return middle_pos
        if mode == "random":
            return int(self.rng.integers(0, n_valid_windows))
        raise ValueError(f"Unknown window mode: {mode}")

    def _select_window_positions(self, n_valid_windows):
        if n_valid_windows <= 0:
            return []
        low_pos = self._single_window_position(n_valid_windows, "low")
        high_pos = self._single_window_position(n_valid_windows, "high")
        middle_pos = self._single_window_position(n_valid_windows, "middle")
        if self.split == "train":
            return [("low", low_pos), ("middle", middle_pos), ("high", high_pos)]
        if self.split == "val":
            pos = self._single_window_position(n_valid_windows, self.val_window_mode)
            return [(self.val_window_mode, pos)]
        if self.split == "test":
            pos = self._single_window_position(n_valid_windows, self.test_window_mode)
            return [(self.test_window_mode, pos)]
        raise ValueError(f"Unsupported split: {self.split}")

    def _load_and_flatten_data(self, file_path):
        data = loadmat(file_path)
        fname = os.path.basename(file_path)
        if "1C" in fname:
            file_source = "1C"
        elif "2C" in fname:
            file_source = "2C"
        elif "3C" in fname:
            file_source = "3C"
        else:
            file_source = "unknown"
        gaf_cell = data.get("GAF_cell", None)
        frag_cell = data.get("FRAG_cell", None)
        cond_cell = data.get("COND_VEC_cell", None)
        info_cell = data.get("info_cell", None)
        seg_idx_cell = data.get("SEG_idx_cell", None)
        if any((x is None for x in [gaf_cell, frag_cell, cond_cell, info_cell])):
            print(
                f"Warning: {file_path} is missing GAF/FRAG/COND_VEC/info fields; skipped"
            )
            return
        n_samples = gaf_cell.shape[0]
        n_segments = frag_cell.shape[1]
        self.total_cycles += n_samples
        for i in range(n_samples):
            full_caf = gaf_cell[i, 0]
            if full_caf.size == 0:
                continue
            cycle_info = self._parse_cycle_info(info_cell[i, 0], i + 1)
            valid_windows = []
            for s in range(n_segments):
                frag = frag_cell[i, s]
                cond = cond_cell[i, s]
                if not self._is_valid_window(frag, cond):
                    self.skipped_empty_segments += 1
                    continue
                segment_idx = self._resolve_segment_idx(seg_idx_cell, i, s)
                cond_vec_raw = self._to_1d_float_vec(cond)
                self.all_cond_vecs_raw.append(cond_vec_raw)
                valid_windows.append(
                    {
                        "full_caf": full_caf,
                        "frag": frag,
                        "cond_vec_raw": cond_vec_raw,
                        "info": cycle_info,
                        "segment_idx": segment_idx,
                        "file_source": file_source,
                    }
                )
            self.valid_segments += len(valid_windows)
            if len(valid_windows) == 0:
                continue
            selected = self._select_window_positions(len(valid_windows))
            for window_role, pos in selected:
                rec = valid_windows[pos]
                self.flat_samples.append({**rec, "window_role": window_role})
                self.selected_segments += 1

    def __len__(self):
        return len(self.flat_samples)

    def __getitem__(self, idx):
        sample = self.flat_samples[idx]
        full_caf = sample["full_caf"]
        frag = sample["frag"]
        cond_vec_raw = sample["cond_vec_raw"]
        full_caf = full_caf.astype(np.float32) / 255.0
        frag = frag.astype(np.float32) / 255.0
        input_tensor = frag[np.newaxis, :, :]
        target_tensor = full_caf[np.newaxis, :, :]
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        input_tensor = torch.from_numpy(input_tensor).float() * 2.0 - 1.0
        target_tensor = torch.from_numpy(target_tensor).float() * 2.0 - 1.0
        cond_vec = (cond_vec_raw - self.cond_mean) / self.cond_std
        cond_vec = torch.from_numpy(cond_vec).float()
        cond_vec_raw = torch.from_numpy(cond_vec_raw).float()
        return {
            "input": input_tensor,
            "target": target_tensor,
            "cond_vec": cond_vec,
            "cond_vec_raw": cond_vec_raw,
            "info": sample["info"],
            "segment_idx": sample["segment_idx"],
            "window_role": sample["window_role"],
            "file_source": sample["file_source"],
        }


def custom_collate_fn(batch):
    inputs = torch.stack([item["input"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    cond_vecs = torch.stack([item["cond_vec"] for item in batch])
    cond_vecs_raw = torch.stack([item["cond_vec_raw"] for item in batch])
    infos = [item["info"] for item in batch]
    segment_idxs = torch.tensor(
        [item["segment_idx"] for item in batch], dtype=torch.int64
    )
    window_roles = [item.get("window_role", "unknown") for item in batch]
    file_sources = [item.get("file_source", "unknown") for item in batch]
    return {
        "input": inputs,
        "target": targets,
        "cond_vec": cond_vecs,
        "cond_vec_raw": cond_vecs_raw,
        "info": infos,
        "segment_idx": segment_idxs,
        "window_role": window_roles,
        "file_source": file_sources,
    }


def get_dataloaders(
    data_dir,
    batch_size=16,
    num_workers=4,
    val_window_mode="random",
    test_window_mode="middle",
    random_seed=42,
):
    pin = torch.cuda.is_available()
    train_dataset = BatteryCAFDataset(
        data_dir, split="train", transform=None, random_seed=random_seed
    )
    cm, cs = (train_dataset.cond_mean, train_dataset.cond_std)
    val_dataset = BatteryCAFDataset(
        data_dir,
        split="val",
        transform=None,
        cond_mean=cm,
        cond_std=cs,
        val_window_mode=val_window_mode,
        test_window_mode=test_window_mode,
        random_seed=random_seed,
    )
    test_dataset = BatteryCAFDataset(
        data_dir,
        split="test",
        transform=None,
        cond_mean=cm,
        cond_std=cs,
        val_window_mode=val_window_mode,
        test_window_mode=test_window_mode,
        random_seed=random_seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=custom_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=custom_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=custom_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return (train_loader, val_loader, test_loader)


def visualize_sample(sample, save_path=None):
    input_tensor = sample["input"]
    target_tensor = sample["target"]
    segment_idx = sample["segment_idx"]
    window_role = sample.get("window_role", "unknown")
    if torch.is_tensor(input_tensor):
        input_tensor = input_tensor.detach().cpu().numpy()
    if torch.is_tensor(target_tensor):
        target_tensor = target_tensor.detach().cpu().numpy()
    if torch.is_tensor(segment_idx):
        segment_idx = segment_idx.item()
    input_tensor = (input_tensor + 1.0) / 2.0
    target_tensor = (target_tensor + 1.0) / 2.0
    cols = 2
    plt.figure(figsize=(4 * cols, 4))
    ax = plt.subplot(1, cols, 1)
    ax.imshow(input_tensor[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"FRAG ({window_role}, seg{segment_idx})")
    ax.axis("off")
    ax = plt.subplot(1, cols, 2)
    ax.imshow(target_tensor[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title("Target (FULL-CAF)")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_dir = "./data"
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=4)
    cond_mean_path = os.path.join(checkpoint_dir, "cond_mean.npy")
    cond_std_path = os.path.join(checkpoint_dir, "cond_std.npy")
    np.save(cond_mean_path, train_loader.dataset.cond_mean.astype(np.float32))
    np.save(cond_std_path, train_loader.dataset.cond_std.astype(np.float32))
    print(f"Saved: {cond_mean_path}")
    print(f"Saved: {cond_std_path}")
    print(f"cond_mean shape: {train_loader.dataset.cond_mean.shape}")
    print(f"cond_std  shape: {train_loader.dataset.cond_std.shape}")
