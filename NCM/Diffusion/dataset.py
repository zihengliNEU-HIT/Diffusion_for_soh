import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import re
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


class BatteryCAFDataset(Dataset):

    def __init__(
        self, data_dir, split="train", transform=None, cond_mean=None, cond_std=None
    ):
        self.data_dir = data_dir
        self.split = split.lower()
        self.transform = transform
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
        self.flat_samples = []
        for file in self.data_files:
            if os.path.exists(file):
                self._load_and_flatten_data(file)
            else:
                print(f"Warning: {file} does not exist")
        print(f"Loaded {split} with {len(self.flat_samples)} valid samples")
        print(
            f"Stats: total cycles {self.total_cycles}, valid segments {self.valid_segments}, skipped empty segments {self.skipped_empty_segments}"
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

    def _compute_cond_stats(self):
        all_cond_vecs = [sample["cond_vec"] for sample in self.flat_samples]
        all_cond_vecs = np.stack(all_cond_vecs, axis=0)
        self.cond_mean = np.mean(all_cond_vecs, axis=0)
        self.cond_std = np.std(all_cond_vecs, axis=0)
        self.cond_std = np.where(self.cond_std < 1e-08, 1.0, self.cond_std)
        print(
            f"Condition-vector z-score stats [{self.split}]: mean={self.cond_mean}, std={self.cond_std}"
        )

    def _to_1d_float_vec(self, x):
        arr = np.array(x, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = np.array([arr], dtype=np.float32)
        return arr.astype(np.float32)

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
            cycle_info = info_cell[i, 0]
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
                cycle_info = [battery_id, cycle_idx]
            elif isinstance(cycle_info, (list, tuple)):
                cycle_info = [int(cycle_info[0]), int(cycle_info[1])]
            else:
                print(f"Warning: Unknown info format: {type(cycle_info)}")
                cycle_info = [0, i + 1]
            for s in range(n_segments):
                frag = frag_cell[i, s]
                cond = cond_cell[i, s]
                if seg_idx_cell is not None and seg_idx_cell[i, s].size > 0:
                    segment_idx = int(seg_idx_cell[i, s][0, 0])
                else:
                    segment_idx = s
                if frag.size == 0:
                    self.skipped_empty_segments += 1
                    continue
                if cond is None or (isinstance(cond, np.ndarray) and cond.size == 0):
                    self.skipped_empty_segments += 1
                    continue
                self.flat_samples.append(
                    {
                        "full_caf": full_caf,
                        "frag": frag,
                        "cond_vec": self._to_1d_float_vec(cond),
                        "info": cycle_info,
                        "segment_idx": segment_idx,
                        "file_source": file_source,
                    }
                )
                self.valid_segments += 1

    def __len__(self):
        return len(self.flat_samples)

    def __getitem__(self, idx):
        sample = self.flat_samples[idx]
        full_caf = sample["full_caf"]
        frag = sample["frag"]
        cond_vec = sample["cond_vec"]
        full_caf = full_caf.astype(np.float32) / 255
        frag = frag.astype(np.float32) / 255
        input_tensor = frag[np.newaxis, :, :]
        target_tensor = full_caf[np.newaxis, :, :]
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        input_tensor = torch.from_numpy(input_tensor).float() * 2.0 - 1.0
        target_tensor = torch.from_numpy(target_tensor).float() * 2.0 - 1.0
        cond_vec = (cond_vec - self.cond_mean) / self.cond_std
        cond_vec = torch.from_numpy(cond_vec).float()
        return {
            "input": input_tensor,
            "target": target_tensor,
            "cond_vec": cond_vec,
            "info": sample["info"],
            "segment_idx": sample["segment_idx"],
            "file_source": sample["file_source"],
        }


def custom_collate_fn(batch):
    inputs = torch.stack([item["input"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    cond_vecs = torch.stack([item["cond_vec"] for item in batch])
    infos = [item["info"] for item in batch]
    segment_idxs = torch.tensor([item["segment_idx"] for item in batch])
    file_sources = [item.get("file_source", "unknown") for item in batch]
    return {
        "input": inputs,
        "target": targets,
        "cond_vec": cond_vecs,
        "info": infos,
        "segment_idx": segment_idxs,
        "file_source": file_sources,
    }


def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    pin = torch.cuda.is_available()
    train_dataset = BatteryCAFDataset(data_dir, split="train", transform=None)
    cm, cs = (train_dataset.cond_mean, train_dataset.cond_std)
    val_dataset = BatteryCAFDataset(
        data_dir, split="val", transform=None, cond_mean=cm, cond_std=cs
    )
    test_dataset = BatteryCAFDataset(
        data_dir, split="test", transform=None, cond_mean=cm, cond_std=cs
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
    ax.set_title(f"FRAG (seg{segment_idx + 1})")
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
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=4)
    print("\nBatch format check:")
    for batch in train_loader:
        print(f"Input shape:  {batch['input'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Cond vec shape: {batch['cond_vec'].shape}")
        print("Cond vec values:", batch["cond_vec"])
        if len(batch["info"]) > 0:
            print(f"First info: {batch['info'][0]}")
        sample = {
            "input": batch["input"][0],
            "target": batch["target"][0],
            "segment_idx": batch["segment_idx"][0],
        }
        visualize_sample(sample)
        break
