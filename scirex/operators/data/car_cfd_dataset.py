# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform).

"""
Car-CFD dataset — download, reorganization, and training class.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# Handle relative imports for standalone execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from .mesh_datamodule import MeshDataModule
except ImportError:
    from scirex.operators.data.mesh_datamodule import MeshDataModule


ZENODO_RECORD_ID = "13936501"

# 1. DATA PREPARATION

def download_dataset(root: Path):
    """Download and extract Car-CFD from Zenodo with corruption check."""
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    root.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching Zenodo metadata ({ZENODO_RECORD_ID})...")
    with urllib.request.urlopen(url) as response:
        metadata = json.loads(response.read().decode())

    for file_obj in metadata.get("files", []):
        file_url = file_obj["links"]["self"]
        file_name = file_obj["key"]
        file_path = root / file_name
        expected_size = file_obj.get("size", None)

        if file_path.exists() and expected_size:
            if file_path.stat().st_size != expected_size:
                print(f" Corrupted file detected: {file_name}. Deleting.")
                file_path.unlink()

        if not file_path.exists():
            print(f"⬇ Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)

        # Extraction logic
        if not (root / "processed-car-pressure-data").exists():
            print(f" Extracting {file_name}...")
            try:
                if file_name.endswith(".zip"):
                    import zipfile
                    with zipfile.ZipFile(file_path, "r") as z: z.extractall(root)
                elif file_name.endswith((".tar.gz", ".tgz", ".tar")):
                    import tarfile
                    mode = "r:gz" if file_name.endswith("gz") else "r:"
                    with tarfile.open(file_path, mode) as t: t.extractall(root)
            except Exception as e:
                print(f" Extraction failed: {e}")
                if file_path.exists(): file_path.unlink()

def save_ply(vertices, faces, filepath):
    """Simple PLY exporter."""
    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def reorganize_data_into_samples(base_dir: Path):
    """
    Standardizes the data layout so MeshDataModule can find items.
    Moves flat mesh_NNN.ply and press_NNN.npy files into samples/NNN/.
    Also handles conversion from .npz if that's what was downloaded.
    """
    data_dir = base_dir / "data"
    samples_dir = data_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    # ── Choice A: NPZ files ────────────────────────────────────────────────
    npz_files = list(data_dir.glob("*.npz"))
    if npz_files:
        print(f" Converting {len(npz_files)} NPZ files into standardized samples/")
        for i, npz_path in enumerate(npz_files):
            idx = npz_path.stem
            s_dir = samples_dir / idx
            s_dir.mkdir(exist_ok=True)
            data = np.load(npz_path, allow_pickle=True)
            v, p = data["vertices"], data["press"]
            
            # Squeeze extra leading dimensions (e.g. (1, N, 3) -> (N, 3))
            while v.ndim > 2: v = np.squeeze(v, 0)
            while p.ndim > 2: p = np.squeeze(p, 0)
            
            # Standardize p to (N, C)
            if p.ndim == 1: p = p[:, None]
            if p.ndim == 2 and p.shape[0] < p.shape[1]: p = p.T # Ensure (N, C)
            
            np.save(s_dir / "press.npy", p.astype(np.float32))
            if "faces" in data:
                save_ply(v, data["faces"], s_dir / "mesh.ply")
                
            if i % 100 == 0: print(f"  Processed {i}/{len(npz_files)}")
        return

    # ── Choice B: Flat PLY/NPY files ───────────────────────────────────────
    mesh_files = list(data_dir.glob("mesh_*.ply"))
    if mesh_files:
        print(f" Reorganizing {len(mesh_files)} flat files into standardized samples/")
        for i, m_path in enumerate(mesh_files):
            idx = m_path.stem.replace("mesh_", "")
            s_dir = samples_dir / idx
            s_dir.mkdir(exist_ok=True)
            
            # Find and copy pressure
            p_path = data_dir / f"press_{idx}.npy"
            if p_path.exists():
                press = np.load(p_path)
                if press.ndim == 1: press = press[:, None]
                if press.ndim == 2 and press.shape[0] < press.shape[1]: press = press.T
                
                np.save(s_dir / "press.npy", press.astype(np.float32))

            # Copy mesh to samples/ID/mesh.ply
            import shutil
            shutil.copy2(m_path, s_dir / "mesh.ply")
            
            if i % 100 == 0: print(f"  Processed {i}/{len(mesh_files)}")

# 2. TRAINING CLASS

class CarCFDDataset(MeshDataModule):
    """Car-CFD Dataset that automatically handles download and reorganization."""
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        n_train: Optional[int] = 1,
        n_test: Optional[int] = 1,
        query_res: List[int] = [32, 32, 32],
        download: bool = True,
        max_neighbors: int = 64,
        in_gno_radius: float = 0.05,
        out_gno_radius: float = 0.05,
        neighbor_cache_dir: Optional[str] = None,
        use_cache: bool = True,
        mesh_backend: str = "trimesh",
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        if download:
            download_dataset(self.root_dir)
            
        # Resolve path to the extracted data
        # Some Zenodo zips might have nested folders or flat files.
        possibilities = [
            self.root_dir / "processed-car-pressure-data/processed-car-pressure-data",
            self.root_dir / "processed-car-pressure-data",
            self.root_dir
        ]
        base = self.root_dir
        for p in possibilities:
            if (p / "data").exists() or (p / "train.txt").exists():
                base = p
                break

        # Ensure we have standardized samples/ directory
        samples_root = base / "data/samples"
        is_populated = samples_root.exists() and any(samples_root.iterdir())
        
        if not is_populated:
            print(f"Standardized samples not found or empty at {samples_root}. Starting reorganization...")
            reorganize_data_into_samples(base)

        super().__init__(
            root_dir=base,
            item_dir_name="samples/",
            n_train=n_train,
            n_test=n_test,
            query_res=query_res,
            attributes=["press"],
            max_neighbors=max_neighbors,
            in_gno_radius=in_gno_radius,
            out_gno_radius=out_gno_radius,
            neighbor_cache_dir=neighbor_cache_dir,
            use_cache=use_cache,
            mesh_backend=mesh_backend,
        )

        # 4. Trimming logic (Must happen AFTER normalization in MeshDataModule for parity)
        # NeuralOperator normalizes on the full 3682 points, then crops to 3586 vertices.
        for it in self.data:
            if "press" in it:
                p = it["press"]
                # p has shape (N, 1) or (1, N). NeuralOp uses (1, 3682). 
                # SciREX standardizes to (N, 1) in MeshDataModule.
                if p.shape[0] > 112:
                    it["press"] = np.concatenate((p[:16, :], p[112:, :]), axis=0)

if __name__ == "__main__":
    root = Path("/home/gazania/zan_folder/SciREX/scirex/operators/data/car_cfd_data")
    ds = CarCFDDataset(root_dir=root, n_train=1, n_test=0)
    print(f"\n✓ Successfully loaded training sample.")
    print(f"Keys: {list(ds.train_data[0].keys())}")
    print(f"Pressure shape: {ds.train_data[0]['press'].shape} (Corrected for parity)")
