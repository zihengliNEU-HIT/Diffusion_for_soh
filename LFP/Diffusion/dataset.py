"""
dataset.py
Content : BatteryCAFDataset - loads *_sliding.mat files, builds flat sample list,
          z-score normalises cond_vec, optionally oversamples late-life cycles.
          get_dataloaders() returns train / val / test DataLoaders.
Run     : python dataset.py   (runs a quick smoke-test)
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from collections import defaultdict


class BatteryCAFDataset(Dataset):

    def __init__(self, data_dir, split="train", transform=None,
                 cond_mean=None, cond_std=None,
                 oversample_tail=False, oversample_tail_factor=3):

        self.data_dir  = data_dir
        self.split     = split.lower()
        self.transform = transform

        # find matching .mat files
        patterns = {"train": r".*_train_sliding\.mat$",
                    "val":   r".*_val_sliding\.mat$",
                    "test":  r".*_test_sliding\.mat$"}
        pat = re.compile(patterns[self.split], re.I)
        self.data_files = [os.path.join(data_dir, f)
                           for f in os.listdir(data_dir) if pat.match(f)]
        if not self.data_files:
            print(f"[Warning] No *_{self.split}_sliding.mat found in {data_dir}")

        self.total_cycles         = 0
        self.skipped_empty_segments = 0
        self.valid_segments       = 0
        self.flat_samples         = []

        for fp in self.data_files:
            if os.path.exists(fp):
                self._load_and_flatten(fp)

        print(f"[{split}] {len(self.flat_samples)} samples  "
              f"(cycles={self.total_cycles}, skipped={self.skipped_empty_segments})")

        # tail oversampling - only for training
        if self.split == "train" and oversample_tail:
            self._oversample_tail_cycles(factor=oversample_tail_factor, tail_ratio=0.20)

        # cond_vec normalisation
        if len(self.flat_samples) > 0:
            if cond_mean is not None and cond_std is not None:
                self.cond_mean = np.asarray(cond_mean, dtype=np.float32)
                self.cond_std  = np.asarray(cond_std,  dtype=np.float32)
                self.cond_std  = np.where(self.cond_std < 1e-8, 1.0, self.cond_std)
            else:
                self._compute_cond_stats()

    # -- normalisation ---------------------------------------------------------

    def _compute_cond_stats(self):
        vecs = np.stack([s["cond_vec"] for s in self.flat_samples], axis=0)
        self.cond_mean = np.mean(vecs, axis=0)
        self.cond_std  = np.std(vecs,  axis=0)
        self.cond_std  = np.where(self.cond_std < 1e-8, 1.0, self.cond_std)

    # -- tail oversampling -----------------------------------------------------

    def _oversample_tail_cycles(self, factor=3, tail_ratio=0.20):
        """Repeat tail-cycle samples (factor-1) times to counteract class imbalance."""
        battery_cycles = defaultdict(list)
        for idx, s in enumerate(self.flat_samples):
            key = (s["file_source"], s["info"][0])
            battery_cycles[key].append((idx, s["info"][1]))

        tail_indices = set()
        for key, pairs in battery_cycles.items():
            cycles    = [c for _, c in pairs]
            lo, hi    = min(cycles), max(cycles)
            threshold = lo + (hi - lo) * (1.0 - tail_ratio)
            for flat_idx, cyc in pairs:
                if cyc >= threshold:
                    tail_indices.add(flat_idx)

        tail_samples = [self.flat_samples[i] for i in sorted(tail_indices)]
        for _ in range(factor - 1):
            self.flat_samples.extend(tail_samples)

        print(f"[TailOversample] {len(tail_indices)} tail samples x{factor}  "
              f"-> total {len(self.flat_samples)}")

    # -- mat loading -----------------------------------------------------------

    @staticmethod
    def _to_1d_float32(x):
        arr = np.squeeze(np.array(x, dtype=np.float32))
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr.astype(np.float32)

    @staticmethod
    def _file_source(fname):
        for tag in ("53C_54", "56C_19", "56C_36"):
            if tag in fname:
                return tag
        return "unknown"

    def _load_and_flatten(self, file_path):
        data       = loadmat(file_path)
        fname      = os.path.basename(file_path)
        file_src   = self._file_source(fname)

        gaf_cell  = data.get("GAF_cell",     None)
        frag_cell = data.get("FRAG_cell",    None)
        cond_cell = data.get("COND_VEC_cell",None)
        info_cell = data.get("info_cell",    None)
        seg_cell  = data.get("SEG_idx_cell", None)

        if any(x is None for x in [gaf_cell, frag_cell, cond_cell, info_cell]):
            print(f"[Warning] Missing fields in {file_path}, skipping.")
            return

        n_cycles, n_segs = gaf_cell.shape[0], frag_cell.shape[1]
        self.total_cycles += n_cycles

        for i in range(n_cycles):
            full_caf = gaf_cell[i, 0]
            if full_caf.size == 0:
                continue

            # parse info -> [battery_id, cycle_idx]
            ci = info_cell[i, 0]
            if isinstance(ci, np.ndarray):
                flat = ci.flatten()
                battery_id, cycle_idx = int(flat[0]), int(flat[1])
            else:
                battery_id, cycle_idx = int(ci[0]), int(ci[1])
            cycle_info = [battery_id, cycle_idx]

            for s in range(n_segs):
                frag = frag_cell[i, s]
                cond = cond_cell[i, s]
                seg_idx = int(seg_cell[i, s][0, 0]) if (seg_cell is not None and seg_cell[i, s].size > 0) else s

                if frag.size == 0 or cond is None or (isinstance(cond, np.ndarray) and cond.size == 0):
                    self.skipped_empty_segments += 1
                    continue

                self.flat_samples.append({
                    "full_caf":    full_caf,
                    "frag":        frag,
                    "cond_vec":    self._to_1d_float32(cond),
                    "segment_idx": seg_idx,
                    "info":        cycle_info,
                    "file_source": file_src,
                })
                self.valid_segments += 1

    # -- Dataset interface -----------------------------------------------------

    def __len__(self):
        return len(self.flat_samples)

    def __getitem__(self, idx):
        s        = self.flat_samples[idx]
        full_caf = s["full_caf"].astype(np.float32) / 255.0
        frag     = s["frag"].astype(np.float32)     / 255.0

        inp = torch.from_numpy(frag[np.newaxis]).float()     * 2.0 - 1.0   # FRAG  [1,H,W]
        tgt = torch.from_numpy(full_caf[np.newaxis]).float() * 2.0 - 1.0   # FULL  [1,H,W]

        cond_vec = (s["cond_vec"] - self.cond_mean) / self.cond_std
        cond_vec = torch.from_numpy(cond_vec).float()

        return {
            "input":       inp,
            "target":      tgt,
            "cond_vec":    cond_vec,
            "info":        s["info"],
            "segment_idx": s["segment_idx"],
            "file_source": s["file_source"],
        }


# -- collate -------------------------------------------------------------------

def custom_collate_fn(batch):
    return {
        "input":       torch.stack([b["input"]    for b in batch]),
        "target":      torch.stack([b["target"]   for b in batch]),
        "cond_vec":    torch.stack([b["cond_vec"] for b in batch]),
        "info":        [b["info"]        for b in batch],
        "segment_idx": torch.tensor([b["segment_idx"] for b in batch]),
        "file_source": [b["file_source"] for b in batch],
    }


# -- dataloaders ---------------------------------------------------------------

def get_dataloaders(data_dir, batch_size=16, num_workers=4,
                    oversample_tail=False, oversample_tail_factor=3):
    pin = torch.cuda.is_available()

    train_ds = BatteryCAFDataset(data_dir, split="train",
                                 oversample_tail=oversample_tail,
                                 oversample_tail_factor=oversample_tail_factor)
    cm, cs   = train_ds.cond_mean, train_ds.cond_std

    val_ds   = BatteryCAFDataset(data_dir, split="val",  cond_mean=cm, cond_std=cs)
    test_ds  = BatteryCAFDataset(data_dir, split="test", cond_mean=cm, cond_std=cs)

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin,
                          collate_fn=custom_collate_fn,
                          persistent_workers=(num_workers > 0),
                          prefetch_factor=(4 if num_workers > 0 and shuffle else
                                           2 if num_workers > 0 else None))

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)


# -- smoke test ----------------------------------------------------------------

if __name__ == "__main__":
    tl, vl, _ = get_dataloaders("./data", batch_size=4)
    batch = next(iter(tl))
    print("input  :", batch["input"].shape)
    print("target :", batch["target"].shape)
    print("cond   :", batch["cond_vec"].shape)
