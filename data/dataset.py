from ast import main
import os
import json
import numpy as np
from typing import List
from pathlib import Path
import scanpy as sc
import h5py

from sympy import false
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial import KDTree
from functools import partial
import math



def normalize_coords_robust(coords, target_range=100.0):
    
    mean = coords.mean(axis=0, keepdims=True) 
    centered = coords - mean
    
    max_val = np.max(np.abs(centered))
    
    norm_coords = centered / (max_val + 1e-6)

    norm_coords = norm_coords * (target_range / 2.0)
    
    return norm_coords


class RegionLevelSTDataset(Dataset):
 

    def __init__(self,
                 sample_ids: List[str],
                 embed_root: str,
                 st_root: str,
                 gene_list_path: str,
                 normalize_method=None,
                 distribution: str = "beta_3_1",
                 sample_times: int = 5,
                ):
        super().__init__()

        self.sample_times = sample_times
        self.region_sampler = RegionSampler(distribution)
        self.slides = []

        # --- load gene names ---
        with open(gene_list_path, 'r') as f:
            self.genes = json.load(f)['genes'][:2000]


        for sid in sample_ids:
            embed_path = os.path.join(embed_root, f"{sid}.h5")
            st_path = os.path.join(st_root, f"{sid}.h5ad")

            # read embeddings and coordinates
            with h5py.File(embed_path, 'r') as f:
                embeddings = f["embeddings"][:]
                coords = f["coords"][:]
                barcodes = f["barcodes"][:].astype(str).flatten().tolist()

            # read expression data
            adata = sc.read_h5ad(st_path)
            adata = adata[barcodes, :][:, self.genes].copy()

            # normalization
            if normalize_method:
                adata = normalize_method(adata)

            gene_exp = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

            # center coordinates for numerical stability
            coords = normalize_coords_robust(coords)
            
            self.slides.append({
                "features": torch.from_numpy(embeddings).float(),
                "genes": torch.from_numpy(gene_exp).float(),
                "coords": torch.from_numpy(coords).float(),
            })

        self.n_chunks = [sample_times] * len(self.slides)

    def __len__(self):
        """Total number of random region samples per epoch."""
        return sum(self.n_chunks)

    def __getitem__(self, idx):
        """Sample one random region from one slide."""
        for i, n_chunk in enumerate(self.n_chunks):
            if idx < n_chunk:
                slide = self.slides[i]
                spot_ids = torch.tensor( self.region_sampler(slide["coords"].numpy()), dtype=torch.long)
                return (
                    slide["features"][spot_ids],
                    slide["coords"][spot_ids],
                    slide["genes"][spot_ids],
                )
            idx -= n_chunk




# Utility: Sampling Distributions
def get_distribution(name: str):
    """Return a callable sampling function given a distribution name."""
    if "constant" in name:
        ratio = float(name.split("_")[1])
        return lambda: ratio
    elif "beta" in name:
        a, b = [float(x) for x in name.split("_")[1:]]
        return partial(np.random.beta, a, b)
    elif name == "uniform":
        return np.random.rand
    elif name == "cosine":
        return lambda: (1 - math.cos(np.random.rand() * math.pi * 0.5))
    elif name == "sqrt":
        return lambda: math.sqrt(np.random.rand())
    elif name == "square":
        return lambda: np.random.rand() ** 2
    else:
        raise ValueError(f"Unknown distribution: {name}")

# Regoin Sampler
class RegionSampler:
    """Sample a local spatial region from given 2D coordinates."""
    def __init__(self, distribution: str = "beta_3_1", min_samples: int = 2):
        self.distribution_func = get_distribution(distribution)
        self.min_samples = min_samples

    def __call__(self, coords: np.ndarray):
        """
        Args:
            coords: [N, 2] array of spatial coordinates.
        Returns:
            indices: (num_samples,) indices of selected nearest patches.
        """
        total_samples = max(self.min_samples, int(len(coords) * self.distribution_func()))
        total_samples = min(total_samples, len(coords))

        if total_samples == len(coords):
            return np.arange(len(coords))

        tree = KDTree(coords)
        center_idx = np.random.randint(0, len(coords))
        _, idx = tree.query(coords[center_idx], k=total_samples)
        return idx

# Collate Function
def region_collate_fn():
    """Pad variable-length patches in a batch to the same size.
    Returns:
        features: [B, N_max, D]
        coords:   [B, N_max, 2]
        genes:   [B, N_max, G]
        masks : [B, N_max]
    """
    def collate(batch):
        feats, coords, genes = zip(*batch)
        max_len = max(f.size(0) for f in feats)
        pad = lambda x: F.pad(x, (0, 0, 0, max_len - x.size(0)))

        feats_padded = []
        coords_padded = []
        genes_padded = []
        masks = []

        for f, c, g in zip(feats, coords, genes):
            n = f.size(0)
            feats_padded.append(pad(f))
            coords_padded.append(pad(c))
            genes_padded.append(pad(g))
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[:n] = True  
            masks.append(mask)

        feats = torch.stack(feats_padded)
        coords = torch.stack(coords_padded)
        genes = torch.stack(genes_padded)
        masks = torch.stack(masks)         # [B, N_max]

        return feats, coords, genes, masks
    return collate
