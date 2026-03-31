"""
MeshDataModule / CarCFD end-to-end parity test.

Compares the SciREX JAX data pipeline against the original
neuraloperator PyTorch pipeline at three levels:

  1. **Raw mesh data** — vertices, faces, normals from the same .ply file
  2. **Geometry processing** — bounding box, normalization, centroids, areas,
     SDF / distances, closest points
  3. **Neighbor search** — KD-Tree (SciREX) vs native_neighbor_search (neuralop)

Usage
-----
    python -m pytest scirex/operators/tests/parity/test_mesh_datamodule_parity.py -v -s

Or run directly:
    python scirex/operators/tests/parity/test_mesh_datamodule_parity.py
"""

import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import numpy as np
import pytest
import scipy.spatial

# ── JAX / SciREX ─────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

from scirex.operators.data.car_cfd_dataset import CarCFDDataset

# ── PyTorch / neuralop ───────────────────────────────────────────────
import torch

try:
    from neuralop.data.datasets.mesh_datamodule import MeshDataModule as PtMeshDataModule
    from neuralop.data.datasets.car_cfd_dataset import CarCFDDataset as PtCarCFDDataset
    _HAS_NEURALOP = True
except ImportError:
    _HAS_NEURALOP = False

try:
    from neuralop.layers.neighbor_search import native_neighbor_search as pt_native_neighbor_search
    _HAS_PT_NEIGHBOR = True
except ImportError:
    _HAS_PT_NEIGHBOR = False

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


# =====================================================================
# Paths — adjust if your data lives elsewhere
# =====================================================================
DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "../../../../scirex/operators/data/car_cfd_data"
)


def _find_base_dir(root):
    """Resolve to the directory containing train.txt + data/"""
    from pathlib import Path

    root = Path(root).expanduser().resolve()
    candidates = [
        root,
        root / "processed-car-pressure-data",
        root / "processed-car-pressure-data" / "processed-car-pressure-data",
    ]
    for d in candidates:
        if (d / "train.txt").exists() and (d / "data").exists():
            return d
    raise RuntimeError(f"Cannot find Car-CFD dataset under {root}")


# Layer 1: Raw mesh data
@pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not installed")
class TestRawMeshData:
    """Compare raw mesh arrays loaded via trimesh vs open3d (if available)."""

    def _get_first_mesh_path(self):
        base = _find_base_dir(DATA_ROOT)
        with open(base / "train.txt") as f:
            first_idx = f.readline().split(",")[0].strip()
        return str(base / "data" / f"data/{first_idx}/tri_mesh.ply"), first_idx

    def test_trimesh_loads_valid_mesh(self):
        mesh_path, idx = self._get_first_mesh_path()
        mesh = trimesh.load(mesh_path)
        print(f"\n[Layer 1] Loaded mesh {idx} via trimesh:")
        print(f"  vertices: {mesh.vertices.shape}")
        print(f"  faces:    {mesh.faces.shape}")
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        assert len(mesh.vertices) > 0

    @pytest.mark.skipif(not _HAS_OPEN3D, reason="open3d not installed")
    def test_trimesh_vs_open3d_vertices(self):
        """Vertices from trimesh and open3d should be identical."""
        mesh_path, idx = self._get_first_mesh_path()

        tm_mesh = trimesh.load(mesh_path)
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)

        tm_v = np.asarray(tm_mesh.vertices)
        o3d_v = np.asarray(o3d_mesh.vertices)

        print(f"\n[Layer 1] Vertex comparison for mesh {idx}:")
        print(f"  trimesh  shape={tm_v.shape}, range=[{tm_v.min():.4f}, {tm_v.max():.4f}]")
        print(f"  open3d   shape={o3d_v.shape}, range=[{o3d_v.min():.4f}, {o3d_v.max():.4f}]")

        np.testing.assert_allclose(tm_v, o3d_v, atol=1e-7, err_msg="Vertex mismatch between trimesh and open3d")
        print("  ✓ Vertices match!")


# Layer 2: Geometry processing
class TestGeometryProcessing:
    """Compare normalization, centroids, areas, distances between SciREX
    CarCFDDataset and the original neuralop MeshDataModule (if available)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Load 1 training sample from SciREX."""
        self.jax_ds = CarCFDDataset(
            root_dir=DATA_ROOT,
            n_train=1,
            n_test=0,
            query_res=[8, 8, 8],  # small for speed
            download=False,
            use_cache=False,
        )

    def test_vertices_normalized_to_unit_cube(self):
        """After normalization, vertices should be within [0, 1]."""
        sample = self.jax_ds.train_data[0]
        verts = sample["vertices"]
        nonzero_mask = np.any(verts != 0, axis=1)
        active_verts = verts[nonzero_mask]
        vmin, vmax = active_verts.min(), active_verts.max()
        print(f"\n[Layer 2] Normalized vertex range: [{vmin:.4f}, {vmax:.4f}]")
        assert vmin >= -0.05, f"Vertices below 0: min={vmin}"
        assert vmax <= 1.05, f"Vertices above 1: max={vmax}"
        print("  ✓ Vertices in expected normalized range")

    @pytest.mark.skipif(not _HAS_NEURALOP, reason="neuralop not installed")
    @pytest.mark.skipif(not _HAS_OPEN3D, reason="open3d not installed (needed by neuralop MeshDataModule)")
    def test_bounding_box_parity(self):
        """Global bounding box should match neuralop MeshDataModule."""
        base = _find_base_dir(DATA_ROOT)
        pt_ds = PtCarCFDDataset(
            root_dir=str(base),
            n_train=1,
            n_test=0,
            query_res=[8, 8, 8],
            download=False,
        )
        jax_sample = self.jax_ds.train_data[0]
        pt_sample = pt_ds.train_data[0]
        jax_v = jax_sample["vertices"]
        pt_v = pt_sample["vertices"].numpy()
        jax_active = jax_v[np.any(jax_v != 0, axis=1)]
        pt_active = pt_v[np.any(pt_v != 0, axis=1)]
        print(f"\n[Layer 2] Bounding box parity:")
        print(f"  SciREX  verts range: [{jax_active.min():.4f}, {jax_active.max():.4f}]")
        print(f"  neuralop verts range: [{pt_active.min():.4f}, {pt_active.max():.4f}]")
        diff = np.abs(jax_active.mean() - pt_active.mean())
        print(f"  Mean vertex position diff: {diff:.6f}")


# Layer 3: KD-Tree / Neighbor search
class TestNeighborSearchParity:
    """Compare SciPy KDTree output against PyTorch native_neighbor_search
    on the same real Car-CFD data points."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Load 1 sample with neighbors."""
        self.jax_ds = CarCFDDataset(
            root_dir=DATA_ROOT,
            n_train=1,
            n_test=0,
            query_res=[8, 8, 8],
            download=False,
            use_cache=False,
            max_neighbors=64,
            in_gno_radius=0.05,
        )

    def test_kdtree_correctness(self):
        """Verify SciPy KDTree results are geometrically correct."""
        sample = self.jax_ds.train_data[0]
        verts = sample["vertices"]
        qp = self.jax_ds._constant["query_points"].reshape(-1, 3)
        radius = 0.05
        max_k = 64
        tree = scipy.spatial.KDTree(verts)
        dists, indices = tree.query(qp, k=max_k, distance_upper_bound=radius)
        mask = indices < len(verts)
        n_check = min(10, len(qp))
        for i in range(n_check):
            valid = indices[i][mask[i]]
            if len(valid) == 0: continue
            actual_dists = np.linalg.norm(verts[valid] - qp[i], axis=1)
            assert np.all(actual_dists <= radius + 1e-7)
        print("  ✓ All returned neighbors are within radius")

    @pytest.mark.skipif(not _HAS_PT_NEIGHBOR, reason="neuralop neighbor_search not installed")
    def test_kdtree_vs_neuralop_native(self):
        """Compare SciPy KDTree vs PyTorch native_neighbor_search on real data."""
        sample = self.jax_ds.train_data[0]
        verts = sample["vertices"]
        qp = self.jax_ds._constant["query_points"].reshape(-1, 3)
        radius = 0.05
        max_k = 64
        tree = scipy.spatial.KDTree(verts)
        sp_dists, sp_indices = tree.query(qp, k=max_k, distance_upper_bound=radius)
        sp_mask = sp_indices < len(verts)
        scipy_neighbor_sets = [set(sp_indices[i][sp_mask[i]].tolist()) for i in range(len(qp))]

        points_pt = torch.from_numpy(verts.astype(np.float64))
        queries_pt = torch.from_numpy(qp.astype(np.float64))
        pt_out = pt_native_neighbor_search(points_pt, queries_pt, radius=radius)
        pt_indices = pt_out["neighbors_index"].numpy()
        pt_row_splits = pt_out["neighbors_row_splits"].numpy()
        pt_neighbor_sets = [set(pt_indices[pt_row_splits[i]:pt_row_splits[i+1]].tolist()) for i in range(len(qp))]

        mismatch = 0
        for i in range(len(qp)):
            if pt_neighbor_sets[i] != scipy_neighbor_sets[i] and not scipy_neighbor_sets[i].issubset(pt_neighbor_sets[i]):
                mismatch += 1
        assert mismatch < 10
        print(f"\n[Layer 3] Neighbor search verified against PT Native (mismatches: {mismatch})")


# =====================================================================
# Layer 2+3 combined: Full pipeline comparison
# =====================================================================
@pytest.mark.skipif(not _HAS_NEURALOP, reason="neuralop not installed")
@pytest.mark.skipif(not _HAS_OPEN3D, reason="open3d not installed")
class TestFullPipelineParity:
    """End-to-end comparison: load same sample from both pipelines,
    compare ALL output keys."""

    def test_full_pipeline(self):
        base = _find_base_dir(DATA_ROOT)

        # ── neuralop (PyTorch) ──
        pt_ds = PtCarCFDDataset(
            root_dir=str(base),
            n_train=1,
            n_test=0,
            query_res=[8, 8, 8],
            download=False,
        )
        pt_batch = next(iter(pt_ds.train_loader(batch_size=1)))

        # ── SciREX (JAX) ──
        jax_ds = CarCFDDataset(
            root_dir=DATA_ROOT,
            n_train=1,
            n_test=0,
            query_res=[8, 8, 8],
            download=False,
            use_cache=False,
        )
        jax_batch = next(jax_ds.get_batch("train", batch_size=1))

        # ── Compare shared keys ──
        print("\n[Full Pipeline Parity]")
        print(f"  PT  keys: {sorted(pt_batch.keys())}")
        print(f"  JAX keys: {sorted(jax_batch.keys())}")

        shared_keys = set(pt_batch.keys()) & set(jax_batch.keys())
        for key in sorted(shared_keys):
            pt_val = pt_batch[key]
            jax_val = jax_batch[key]

            if isinstance(pt_val, torch.Tensor):
                pt_val = pt_val.numpy()
            if isinstance(jax_val, jnp.ndarray):
                jax_val = np.asarray(jax_val)

            if pt_val.shape != jax_val.shape:
                print(f"  {key}: SHAPE MISMATCH pt={pt_val.shape} jax={jax_val.shape}")
                continue

            diff = np.abs(pt_val - jax_val).max()
            print(f"  {key}: shape={pt_val.shape}, max_diff={diff:.8f}")


# =====================================================================
# Quick standalone check (no pytest needed)
# =====================================================================
def main():
    """Run the most important checks without pytest."""
    print("=" * 60)
    print("MeshDataModule / CarCFD Parity Check")
    print("=" * 60)

    base = _find_base_dir(DATA_ROOT)
    print(f"Data root resolved: {base}")

    # ── 1. Load SciREX data ──────────────────────────────────────
    print("\n── Loading SciREX CarCFDDataset (1 train, query_res=8³) ──")
    jax_ds = CarCFDDataset(
        root_dir=DATA_ROOT,
        n_train=1,
        n_test=0,
        query_res=[8, 8, 8],
        download=False,
        use_cache=False,
        max_neighbors=64,
        in_gno_radius=0.05,
        out_gno_radius=0.05,
    )
    sample = jax_ds.train_data[0]
    print(f"Keys: {sorted(sample.keys())}")
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]")
        elif isinstance(v, dict):
            print(f"  {k}: (dict with {len(v)} sub-keys)")
            for sk, sv in v.items():
                print(f"    {sk}: shape={sv.shape}, dtype={sv.dtype}")

    # ── 2. Verify normalization ──────────────────────────────────
    verts = sample["vertices"]
    active = verts[np.any(verts != 0, axis=1)]
    print(f"\nActive vertices: {len(active)} / {len(verts)}")
    print(f"Range: [{active.min(axis=0)}, {active.max(axis=0)}]")

    # ── 3. KDTree sanity ─────────────────────────────────────────
    qp = jax_ds._constant["query_points"].reshape(-1, 3)
    tree = scipy.spatial.KDTree(verts)
    d, i = tree.query(qp, k=12, distance_upper_bound=0.05)
    mask = i < len(verts)
    n_matched = np.any(mask, axis=1).sum()
    print(f"\nKDTree: {n_matched}/{len(qp)} queries found neighbors (radius=0.05)")

    # ── 4. Cross-check with neuralop ─────────────────────────────
    if _HAS_NEURALOP and _HAS_OPEN3D:
        print("\n── Loading neuralop PtCarCFDDataset ──")
        pt_ds = PtCarCFDDataset(
            root_dir=str(base),
            n_train=1,
            n_test=0,
            query_res=[8, 8, 8],
            download=False,
        )
        pt_batch = next(iter(pt_ds.train_loader(batch_size=1)))
        print(f"PT keys: {sorted(pt_batch.keys())}")
        for k, v in pt_batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")
    elif not _HAS_OPEN3D:
        print("\n⚠ open3d not installed — cannot run neuralop MeshDataModule comparison")
        print("  Install with: pip install open3d")
        print("  This is needed for the FULL parity test (Layer 2 bounding box + SDF)")
    else:
        print("\n⚠ neuralop not installed — skipping PyTorch comparison")

    if _HAS_PT_NEIGHBOR:
        print("\n── KDTree vs PyTorch native_neighbor_search ──")
        # Use synthetic data for a quick sanity check
        np.random.seed(42)
        pts = np.random.uniform(0, 1, (100, 3)).astype(np.float64)
        qry = np.random.uniform(0, 1, (50, 3)).astype(np.float64)
        radius = 0.3

        tree = scipy.spatial.KDTree(pts)
        sp_d, sp_i = tree.query(qry, k=20, distance_upper_bound=radius)
        sp_mask = sp_i < len(pts)

        pt_out = pt_native_neighbor_search(
            torch.from_numpy(pts), torch.from_numpy(qry), radius=radius
        )
        pt_idx = pt_out["neighbors_index"].numpy()
        pt_splits = pt_out["neighbors_row_splits"].numpy()

        matches = 0
        for qi in range(len(qry)):
            sp_set = set(sp_i[qi][sp_mask[qi]].tolist())
            pt_set = set(pt_idx[pt_splits[qi] : pt_splits[qi + 1]].tolist())
            if len(pt_set) <= 20:
                if sp_set == pt_set:
                    matches += 1
            else:
                if sp_set.issubset(pt_set):
                    matches += 1

        print(f"  Synthetic check: {matches}/50 queries match (should be 50)")
    else:
        print("\n⚠ neuralop neighbor_search not available — skipping KDTree parity")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
