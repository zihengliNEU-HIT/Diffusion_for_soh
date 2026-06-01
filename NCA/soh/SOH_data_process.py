"""
SOH_data_process.py (v2 - image-only)
----------------------------------------------------------------------------
Expected MAT fields for each split:

  train:
    GAF_real  [N, H, W]        values in [0, 1]
    GAF_gen   [N, 60, H, W]    values in [0, 1], 3 windows x 20 images
    CAPACITY  [N, 1]
    INFO      [N, 2]

  val:
    GAF_gen   [N, 20, H, W]    values in [0, 1], one selected window
    CAPACITY  [N, 1]
    INFO      [N, 2]

  test:
    GAF_gen   [N, K, H, W]     K=1 or 5
    CAPACITY  [N, 1]
    INFO      [N, 2]

COND_VEC is not used.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


def resolve_split_dir(data_dir, split):
    """Resolve Diffusion-style soh_input/<split> folders."""
    split = split.lower()
    pat = re.compile(rf'.*_{split}_soh\.mat$', re.I)
    if os.path.isdir(data_dir) and any(pat.match(f) for f in os.listdir(data_dir)):
        return data_dir

    candidates = {
        'train': ('train',),
        'val': ('val',),
        'test': ('test_random', 'test_low', 'test_middle', 'test_high'),
    }.get(split, (split,))

    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isdir(path) and any(pat.match(f) for f in os.listdir(path)):
            return path

    return data_dir


class SOHDataset(Dataset):
    """
    __getitem__ returns:
        gaf_gen   : Tensor [K, 1, H, W]  [-1, 1]
        soh       : Tensor [1]           min-max normalized
        battery_id: int
        cycle_idx : int
        gaf_real  : Tensor [1, H, W]     present only for train samples
    """

    def __init__(self, data_dir, split,
                 soh_min=None, soh_max=None):
        self.split = split.lower()
        self.samples = []
        data_dir = resolve_split_dir(data_dir, self.split)

        pat = re.compile(rf'.*_{self.split}_soh\.mat$', re.I)
        files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if pat.match(f)
        ])
        if not files:
            raise FileNotFoundError(
                f"No '*_{self.split}_soh.mat' found in {data_dir}"
            )

        # for fpath in files:
        #     self._load_mat(fpath)

        for file_idx, fpath in enumerate(files):    
            self._load_mat(fpath, file_idx)

        print(f"[{split}] Loaded {len(self.samples)} samples from {len(files)} file(s)")

        # SOH Min-Max
        if soh_min is not None:
            self.soh_min = float(soh_min)
            self.soh_max = float(soh_max)
        else:
            self._compute_soh_stats()

        if abs(self.soh_max - self.soh_min) < 1e-8:
            self.soh_max = self.soh_min + 1.0

        print(f"[{split}] SOH Min={self.soh_min:.4f} Ah  Max={self.soh_max:.4f} Ah")

    # def _load_mat(self, fpath):
    #     mat = loadmat(fpath)

    #     gaf_gen  = mat['GAF_gen'].astype(np.float32)          # [N, K, H, W]
    #     capacity = mat['CAPACITY'].astype(np.float32).reshape(-1)
    #     info     = mat['INFO'].astype(np.int32)                # [N, 2]

    #     has_real = 'GAF_real' in mat
    #     gaf_real = mat['GAF_real'].astype(np.float32) if has_real else None

    #     N = gaf_gen.shape[0]
    #     for i in range(N):
    #         sample = {
    #             'gaf_gen':    gaf_gen[i],       # [K, H, W]
    #             'capacity':   float(capacity[i]),
    #             'battery_id': int(info[i, 0]),
    #             'cycle_idx':  int(info[i, 1]),
    #         }
    #         if gaf_real is not None:
    #             sample['gaf_real'] = gaf_real[i]  # [H, W]
    #         self.samples.append(sample)

    def _load_mat(self, fpath, file_idx):
        mat = loadmat(fpath)

        gaf_gen  = mat['GAF_gen'].astype(np.float32)
        capacity = mat['CAPACITY'].astype(np.float32).reshape(-1)
        info     = mat['INFO'].astype(np.int32)

        has_real = 'GAF_real' in mat
        gaf_real = mat['GAF_real'].astype(np.float32) if has_real else None

        N = gaf_gen.shape[0]
        for i in range(N):
         
            unique_bid = file_idx * 100000 + int(info[i, 0])

            sample = {
                'gaf_gen':    gaf_gen[i],
                'capacity':   float(capacity[i]),
                'battery_id': unique_bid,         
                'cycle_idx':  int(info[i, 1]),
                'file_name':  os.path.basename(fpath),   
            }
            if gaf_real is not None:
                sample['gaf_real'] = gaf_real[i]
            self.samples.append(sample)

    def _compute_soh_stats(self):
        caps = np.array([s['capacity'] for s in self.samples])
        self.soh_min = float(caps.min())
        self.soh_max = float(caps.max())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # GAF_gen: [K, H, W] -> [K, 1, H, W]
        gaf_gen = torch.from_numpy(s['gaf_gen']).unsqueeze(1)  # [K,1,H,W]
        gaf_gen = gaf_gen * 2.0 - 1.0

        # SOH Min-Max
        soh = (s['capacity'] - self.soh_min) / (self.soh_max - self.soh_min)
        soh = torch.tensor([soh], dtype=torch.float32)

        out = {
            'gaf_gen':    gaf_gen,
            'soh':        soh,
            'battery_id': s['battery_id'],
            'cycle_idx':  s['cycle_idx'],
        }

        if 'gaf_real' in s:
            gaf_real = torch.from_numpy(s['gaf_real']).unsqueeze(0)  # [1,H,W]
            gaf_real = gaf_real * 2.0 - 1.0
            out['gaf_real'] = gaf_real

        return out


def collate_fn(batch):
    out = {
        'gaf_gen':    torch.stack([b['gaf_gen'] for b in batch]),  # [B,K,1,H,W]
        'soh':        torch.stack([b['soh']     for b in batch]),
        'battery_id': [b['battery_id'] for b in batch],
        'cycle_idx':  [b['cycle_idx']  for b in batch],
    }
    if 'gaf_real' in batch[0]:
        out['gaf_real'] = torch.stack([b['gaf_real'] for b in batch])  # [B,1,H,W]
    return out


def get_dataloaders(cfg):
    """
    Return (train_loader, val_loader, test_loader, scaler_dict).

    scaler_dict: soh_min / soh_max
    """
    pin = torch.cuda.is_available()

    train_ds = SOHDataset(cfg.data_dir, 'train')
    val_ds   = SOHDataset(cfg.data_dir, 'val',
                          soh_min=train_ds.soh_min, soh_max=train_ds.soh_max)
    test_ds  = SOHDataset(cfg.data_dir, 'test',
                          soh_min=train_ds.soh_min, soh_max=train_ds.soh_max)

    scaler = {
        'soh_min': train_ds.soh_min,
        'soh_max': train_ds.soh_max,
    }

    kw = dict(num_workers=cfg.num_workers, pin_memory=pin,
              collate_fn=collate_fn,
              persistent_workers=(cfg.num_workers > 0))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,
                              prefetch_factor=4 if cfg.num_workers > 0 else None,
                              **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False,
                              prefetch_factor=2 if cfg.num_workers > 0 else None,
                              **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                              shuffle=False,
                              prefetch_factor=2 if cfg.num_workers > 0 else None,
                              **kw)

    return train_loader, val_loader, test_loader, scaler
