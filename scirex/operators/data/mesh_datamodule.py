# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform).

"""
MeshDataModule — JAX-compatible general dataset for irregular coordinate meshes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.spatial

from ..training.normalizers import GaussianNormalizer

_o3d_available: bool
try:
    import open3d as o3d
    _o3d_available = True
except ModuleNotFoundError:
    _o3d_available = False

_trimesh_available: bool
try:
    import trimesh as _trimesh
    _trimesh_available = True
except ModuleNotFoundError:
    _trimesh_available = False


class MeshDataModule:
    def __init__(
        self,
        root_dir: Union[str, Path],
        item_dir_name: str,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None,
        query_res: List[int] = [32, 32, 32],
        attributes: Optional[List[str]] = None,
        max_neighbors: int = 64,
        in_gno_radius: float = 0.05,
        out_gno_radius: float = 0.05,
        neighbor_cache_dir: Optional[str] = None,
        use_cache: bool = True,
        mesh_backend: str = "trimesh",
        min_b: Optional[np.ndarray] = None,
        max_b: Optional[np.ndarray] = None,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.item_dir_name = item_dir_name
        self.max_neighbors = max_neighbors
        self.in_gno_radius = in_gno_radius
        self.out_gno_radius = out_gno_radius
        self.neighbor_cache_dir = Path(neighbor_cache_dir) if neighbor_cache_dir else None
        self.use_cache = use_cache
        self.attributes = attributes or []
        self._backend = self._resolve_backend(mesh_backend)

        # 1. Read indices
        train_ind = self._read_index_file(self.root_dir / "train.txt")
        test_ind = self._read_index_file(self.root_dir / "test.txt")

        if n_train is not None: train_ind = train_ind[:n_train]
        if n_test is not None: test_ind = test_ind[:n_test]
        
        n_train, n_test = len(train_ind), len(test_ind)
        mesh_ind = train_ind + test_ind
        data_dir = self.root_dir / "data"

        # 2. Load meshes and compute global bounding box
        print(f"Loading {len(mesh_ind)} meshes from {data_dir}...")
        meshes = self._load_meshes(data_dir, mesh_ind)
        
        # Use provided bbox or compute from current selection
        if min_b is not None and max_b is not None:
             self._min_b, self._max_b = min_b, max_b
        else:
             self._min_b, self._max_b = self._get_global_bounding_box(meshes)
        
        min_b, max_b = self._min_b, self._max_b

        # Build shared query grid
        l = [np.linspace(min_b[i], max_b[i], query_res[i]) for i in range(3)]
        query_points = np.stack(np.meshgrid(*l, indexing="ij"), axis=-1).astype(np.float32)
        
        # Shared constant query points normalized to [0, 1]
        self._constant = {
            "query_points": self._range_normalize(query_points, min_b, max_b, 0.0, 1.0).astype(np.float32)
        }

        # 3. Process items
        print(f"Processing geometry and computing distances for {len(meshes)} items...")
        self.data: List[Dict[str, np.ndarray]] = []
        
        # We also want to cache distances, normals, etc. 
        # because computing proximity for 611 items takes ~23 minutes.
        item_cache_dir = None
        if self.neighbor_cache_dir:
            item_cache_dir = self.neighbor_cache_dir / "item_data"
            item_cache_dir.mkdir(parents=True, exist_ok=True)

        for i, mesh in enumerate(meshes):
            idx_str = mesh_ind[i]
            cache_file = item_cache_dir / f"item_{idx_str}.npz" if item_cache_dir else None
            
            if self.use_cache and cache_file and cache_file.exists():
                cached = np.load(cache_file)
                item = {k: cached[k] for k in cached}
            else:
                item = {}
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles if self._backend == "open3d" else mesh.faces)
                
                # 3a. Geometry Normalization [0, 1]
                item["vertices"] = self._range_normalize(verts, min_b, max_b, 0.0, 1.0).astype(np.float32)
                item["centroids"], item["areas"] = self._compute_triangle_stats(verts, faces)
                item["centroids"] = self._range_normalize(item["centroids"], min_b, max_b, 0.0, 1.0).astype(np.float32)

                # 3b. Local Normals
                if self._backend == "trimesh":
                    item["vertex_normals"] = np.asarray(mesh.vertex_normals).astype(np.float32)
                    item["triangle_normals"] = np.asarray(mesh.face_normals).astype(np.float32)
                else: # open3d
                    mesh.compute_vertex_normals()
                    mesh.compute_triangle_normals()
                    item["vertex_normals"] = np.asarray(mesh.vertex_normals).astype(np.float32)
                    item["triangle_normals"] = np.asarray(mesh.triangle_normals).astype(np.float32)

                # 3c. SDF Distance & Closest Points
                dist, closest = self._compute_distances(mesh, query_points)
                item["distance"] = np.expand_dims(dist, -1).astype(np.float32)
                item["closest_points"] = self._range_normalize(closest, min_b, max_b, 0.0, 1.0).astype(np.float32)
                
                if self.use_cache and cache_file:
                    np.savez_compressed(cache_file, **item)
            
            # 3d. Load dynamic attributes (not cached because they might vary)
            for attr in self.attributes:
                p = data_dir / self.item_dir_name / idx_str / f"{attr}.npy"
                if p.exists():
                    # Load and apply strict shape correction to (N, C)
                    val = np.load(p).astype(np.float32)
                    if val.ndim == 1: 
                        val = val[:, None]
                    elif val.ndim == 2 and val.shape[0] < val.shape[1]:
                        val = val.T
                    elif val.ndim > 2:
                        val = val.reshape(val.shape[0], -1) 
                    item[attr] = val
            
            self.data.append(item)
            if (i+1) % 50 == 0: print(f"  Processed {i+1}/{len(meshes)}", flush=True)

        # 4. Attribute-specific Normalization [1e-6, 1] for Areas and Distance
        if n_train > 0:
            train_items = self.data[:n_train]
            d_min = min(it["distance"].min() for it in train_items)
            d_max = max(it["distance"].max() for it in train_items)
            a_min = min(it["areas"].min() for it in train_items)
            a_max = max(it["areas"].max() for it in train_items)
            
            for it in self.data:
                it["distance"] = self._range_normalize(it["distance"], d_min, d_max, 1e-6, 1.0).astype(np.float32)
                it["areas"] = self._range_normalize(it["areas"], a_min, a_max, 1e-6, 1.0).astype(np.float32)
                if it["areas"].ndim == 1: it["areas"] = it["areas"][:, None]

        # 5. Pre-computed Neighbor Search (KD-Tree with Simple Cache)
        print(f"Pre-computing neighbors (KD-Tree)...")
        flat_q_norm = self._constant["query_points"].reshape(-1, 3)
        for i, it in enumerate(self.data):
            # Check for cache
            cache_file = None
            if self.neighbor_cache_dir:
                self.neighbor_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self.neighbor_cache_dir / f"neighbors_{mesh_ind[i]}.npz"
            
            if self.use_cache and cache_file and cache_file.exists():
                cached = np.load(cache_file)
                it["in_neighbors"] = {k.replace("in_", ""): cached[k] for k in cached if k.startswith("in_")}
                it["out_neighbors"] = {k.replace("out_", ""): cached[k] for k in cached if k.startswith("out_")}
            else:
                it["in_neighbors"] = self._kdtree_search(it["vertices"], flat_q_norm, self.in_gno_radius)
                it["out_neighbors"] = self._kdtree_search(flat_q_norm, it["vertices"], self.out_gno_radius)
                
                if self.use_cache and cache_file:
                    save_dict = {}
                    for k, v in it["in_neighbors"].items(): save_dict[f"in_{k}"] = v
                    for k, v in it["out_neighbors"].items(): save_dict[f"out_{k}"] = v
                    np.savez(cache_file, **save_dict)
            
            if (i+1) % 50 == 0: print(f"  Neighbor search {i+1}/{len(meshes)}", flush=True)

        # 6. Final Gaussian Normalization (on pressure etc.)
        self.train_data = self.data[:n_train]
        self.test_data = self.data[n_train:]
        self.normalizers = {}
        
        for attr in self.attributes:
            if n_train > 0 and attr in self.train_data[0]:
                stacked = jnp.stack([jnp.array(d[attr]) for d in self.train_data])
                norm = GaussianNormalizer(stacked)
                self.normalizers[attr] = norm
                for d in self.data:
                    d[attr] = np.array(norm.encode(jnp.array(d[attr])))

    def _resolve_backend(self, preferred):
        if preferred == "open3d" and _o3d_available: return "open3d"
        if _trimesh_available: return "trimesh"
        raise RuntimeError("No mesh backend available.")

    def _read_index_file(self, path):
        with open(path) as f:
            return [x.strip() for x in f.read().replace('\n', '').split(',') if x.strip()]

    def _load_meshes(self, data_dir, indices):
        meshes = []
        for ind in indices:
            p = data_dir / self.item_dir_name / ind / "mesh.ply"
            if not p.exists(): p = data_dir / self.item_dir_name / ind / "tri_mesh.ply"
            
            if not p.exists():
                raise FileNotFoundError(f"Missing mesh for index '{ind}' at: {p.absolute()}\n"
                                        f"Check your reorganization settings.")
            
            meshes.append(o3d.io.read_triangle_mesh(str(p)) if self._backend == "open3d" else _trimesh.load(str(p)))
        return meshes

    def _get_global_bounding_box(self, meshes):
        all_min = [np.asarray(m.vertices).min(axis=0) for m in meshes]
        all_max = [np.asarray(m.vertices).max(axis=0) for m in meshes]
        return np.stack(all_min).min(axis=0), np.stack(all_max).max(axis=0)

    def _compute_triangle_stats(self, verts, faces):
        A, B, C = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        centroids = (A + B + C) / 3.0
        areas = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, axis=1)) / 2.0
        return centroids.astype(np.float32), areas.astype(np.float32)

    def _compute_distances(self, mesh, queries):
        # NeuralOperator forces signed_distance=True for watertight meshes
        if self._backend == "open3d":
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            dist = scene.compute_signed_distance(queries.astype(np.float32)).numpy()
            closest = scene.compute_closest_points(queries.astype(np.float32))["points"].numpy()
        else:
            # Fallback for trimesh
            if mesh.is_watertight:
                from trimesh.proximity import signed_distance
                dist = signed_distance(mesh, queries.reshape(-1, 3))
                dist = dist.reshape(queries.shape[:-1])
            else:
                _, dist, _ = _trimesh.proximity.closest_point(mesh, queries.reshape(-1, 3))
                dist = dist.reshape(queries.shape[:-1])
            
            closest, _, _ = _trimesh.proximity.closest_point(mesh, queries.reshape(-1, 3))
            closest = closest.reshape(queries.shape)
        return dist, closest

    def _range_normalize(self, data, old_min, old_max, new_min, new_max):
        return (data - old_min) / (old_max - old_min + 1e-12) * (new_max - new_min) + new_min

    def _kdtree_search(self, data, queries, radius):
        tree = scipy.spatial.KDTree(data)
        d, i = tree.query(queries, k=self.max_neighbors, distance_upper_bound=radius)
        mask = i < len(data)
        return {
            "neighbor_indices": np.where(mask, i, -1).astype(np.int32),
            "mask": mask.astype(np.float32),
            "distances": np.where(mask, d, 0.0).astype(np.float32)
        }

    def get_batch(self, split="train", batch_size=1, shuffle=True):
        samples = self.train_data if split == "train" else self.test_data
        indices = np.random.permutation(len(samples)) if shuffle else np.arange(len(samples))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[samples[j] for j in batch_indices])
            for k, v in self._constant.items():
                batch[k] = jnp.stack([jnp.array(v)] * len(batch_indices))
            yield batch
