from typing import List, Union, Dict, Optional
from pathlib import Path
import json
import jax
import urllib.request
import numpy as np
import scipy.spatial
import jax.numpy as jnp


def download_from_zenodo_record(record_id: str, root: Path):
    url = f"https://zenodo.org/api/records/{record_id}"

    with urllib.request.urlopen(url) as response:
        metadata = json.loads(response.read().decode())

    root.mkdir(parents=True, exist_ok=True)

    for file_obj in metadata.get("files", []):
        file_url = file_obj["links"]["self"]
        file_name = file_obj["key"]
        file_path = root / file_name

        if not file_path.exists():
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)

            # extract
            if file_name.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(file_path, "r") as z:
                    z.extractall(root)
            elif file_name.endswith((".tar.gz", ".tgz", ".tar")):
                import tarfile
                mode = "r:gz" if file_name.endswith("gz") else "r:"
                with tarfile.open(file_path, mode) as t:
                    t.extractall(root)

class CarCFDDataset:
    def __init__(
        self,
        root_dir: Union[str, Path],
        n_train: int = 1,
        n_test: int = 1,
        query_res: List[int] = [32, 32, 32],
        download: bool = True,
        max_vertices: int = 4096,
        max_neighbors: int = 12,
        in_gno_radius: float = 0.05,
        out_gno_radius: float = 0.05,
        neighbor_cache_dir: Optional[str] = "./scirex/operators/data/neighbor_cache",
        use_cache: bool = True,
        normalize_mesh: bool = True,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.query_res = query_res
        self.max_vertices = max_vertices
        self.zenodo_record_id = "13936501"

        # Neighbor search properties
        self.max_neighbors = max_neighbors
        self.in_gno_radius = in_gno_radius
        self.out_gno_radius = out_gno_radius
        self.use_cache = use_cache
        self.normalize_mesh = normalize_mesh
        self.neighbor_cache_dir = Path(neighbor_cache_dir).expanduser().resolve() if neighbor_cache_dir else None
        
        if self.use_cache and self.neighbor_cache_dir:
            self.neighbor_cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True)

        if download and not (self.root_dir / "processed-car-pressure-data").exists():
            download_from_zenodo_record(self.zenodo_record_id, self.root_dir)

        self.base_dir, self.data_dir = self._find_data_dir()

        self.train_data = self._load_split("train.txt", idx_split="train", n_samples=n_train)
        self.test_data = self._load_split("test.txt", idx_split="test", n_samples=n_test)

    # --------------------------------------------------
    def _find_data_dir(self):
        candidates = [
            self.root_dir,
            self.root_dir / "processed-car-pressure-data",
            self.root_dir / "processed-car-pressure-data" / "processed-car-pressure-data"
        ]

        for d in candidates:
            if (d / "train.txt").exists() and (d / "data").exists():
                return d, d / "data"

        raise RuntimeError("Dataset structure not found")

    # --------------------------------------------------
    def _load_split(self, split_file, idx_split, n_samples):
        path = self.base_dir / split_file

        with open(path) as f:
            indices = [i.strip() for i in f.readline().split(",") if i.strip()]

        data = []
        for idx in indices[:n_samples]:
            item = self._load_item(idx, f"{idx_split}_{idx}")
            if item:
                data.append(item)

        return data

    # --------------------------------------------------
    def _load_item(self, idx: str, cache_idx: str) -> Optional[Dict]:
        """Loads a single item (either .npz or raw .npy/.ply) with padding for JAX."""
        # --- 1. Attempt .npz Load ---
        npz_path = self.data_dir / f"{idx}.npz"
        item = None

        if npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as data:
                item = {k: data[k] for k in data.files}
        else:
            # --- 2. Fallback to Raw PLY/NPY ---
            # Try nested: data/<idx>/press.npy OR flat: press_<idx>.npy
            press_path = self.data_dir / "data" / idx / "press.npy"
            if not press_path.exists():
                press_path = self.data_dir / f"press_{idx}.npy"

            mesh_path = self.data_dir / "data" / idx / "tri_mesh.ply"
            if not mesh_path.exists():
                mesh_path = self.data_dir / f"mesh_{idx}.ply"

            if press_path.exists() and mesh_path.exists():
                try:
                    import trimesh
                    mesh = trimesh.load(mesh_path)
                    vertices = mesh.vertices
                    faces = mesh.faces
                    centroids = mesh.vertices[faces].mean(axis=1)
                    press = np.load(press_path)
                    
                    item = {
                        "vertices": vertices,
                        "press": press,
                        "centroids": centroids
                    }
                except Exception as e:
                    print(f"Error reading raw data for {idx}: {e}")
                    return None

        if item is None:
            return None

        vertices = item["vertices"]
        press = item["press"]
        centroids = item.get("centroids")

        # --- Geometry Normalization ---
        if self.normalize_mesh:
            v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
            center = (v_min + v_max) / 2.0
            max_range = (v_max - v_min).max()
            scale = 0.9 / (max_range + 1e-8)  # Isotropic scale with padding

            vertices = (vertices - center) * scale + 0.5
            item["vertices"] = vertices
            
            if centroids is not None:
                centroids = (centroids - center) * scale + 0.5
                item["centroids"] = centroids
            
            if "distance" in item:
                item["distance"] = item["distance"] * scale

        if press.ndim == 2 and press.shape[1] > 112:
            press = np.concatenate((press[:, 0:16], press[:, 112:]), axis=1)

        if press.ndim == 1:
            press = press[:, None]

        vertices = self._pad(vertices)
        press = self._pad(press)
        if centroids is not None:
            centroids = self._pad(centroids)

        query_points = self._generate_grid()
        
        # --- Distance / SDF Calculation ---
        # 1. Use existing if present in item
        if "distance" in item:
            distance = item["distance"]
        else:
            # 2. Compute using trimesh if possible
            try:
                import trimesh
                # Reconstruct mesh for SDF calculation
                # We need faces to compute proximity properly
                if "faces" in item:
                    faces = item["faces"]
                else:
                    # If faces are missing, proximity calculation won't work well
                    # fallback to zeros or nearest vertex distance
                    faces = None
                
                if faces is not None:
                    mesh = trimesh.Trimesh(vertices=item["vertices"], faces=faces)
                    distance = self._compute_sdf(mesh, query_points)
                else:
                    # Fallback: Approximate Distance Function (distance to nearest vertex)
                    tree = scipy.spatial.KDTree(item["vertices"])
                    d, _ = tree.query(query_points)
                    distance = d[..., None]
            except (ImportError, Exception) as e:
                print(f"Warning: SDF calculation failed for {idx}: {e}. Using zeros.")
                distance = np.zeros((*self.query_res, 1), dtype=np.float32)

        sample = {
            "vertices": vertices.astype(np.float32),
            "press": press.astype(np.float32),
            "query_points": query_points,
            "distance": distance.astype(np.float32),
        }
        if centroids is not None:
            sample["centroids"] = centroids.astype(np.float32)

        # --- Add Neighbors to Sample ---
        if self.use_cache:
            sample = self._add_neighbors(sample, cache_idx)
            
        return sample

    def _add_neighbors(self, sample: Dict, cache_idx: str) -> Dict:
        """Adds pre-calculated neighbors with caching."""
        res_str = "x".join(map(str, self.query_res))
        cache_id = f"{cache_idx}_v{self.max_vertices}_res{res_str}_n{self.max_neighbors}_rin{self.in_gno_radius}_rout{self.out_gno_radius}_norm{self.normalize_mesh}"
        cache_path = self.neighbor_cache_dir / f"{cache_id}.npz"

        if cache_path.exists():
            with np.load(cache_path, allow_pickle=True) as cached:
                sample["in_neighbors"] = {k.replace("in_", ""): cached[k] for k in cached.files if k.startswith("in_")}
                sample["out_neighbors"] = {k.replace("out_", ""): cached[k] for k in cached.files if k.startswith("out_")}
                if "distance" in cached:
                    sample["distance"] = cached["distance"]
            return sample

        # 1. In (Mesh -> Grid)
        in_nb = self._compute_neighbors_kdtree(
            data=sample["vertices"],
            queries=sample["query_points"].reshape(-1, 3),
            radius=self.in_gno_radius
        )

        # 2. Out (Grid -> Mesh)
        # We query at centroids to match face-based pressure targets if available
        out_queries = sample.get("centroids", sample["vertices"])
        out_nb = self._compute_neighbors_kdtree(
            data=sample["query_points"].reshape(-1, 3),
            queries=out_queries,
            radius=self.out_gno_radius
        )

        # Save
        save_dict = {f"in_{k}": v for k, v in in_nb.items()}
        save_dict.update({f"out_{k}": v for k, v in out_nb.items()})
        save_dict["distance"] = sample["distance"]
        np.savez(cache_path, **save_dict)

        sample["in_neighbors"] = in_nb
        sample["out_neighbors"] = out_nb
        return sample

    def _compute_neighbors_kdtree(self, data, queries, radius):
        tree = scipy.spatial.KDTree(data)
        d, i = tree.query(queries, k=self.max_neighbors, distance_upper_bound=radius)
        mask = (i < len(data))
        valid_indices = np.where(mask, i, -1).astype(np.int32)
        d_valid = np.where(mask, d, 0.0).astype(np.float32)
        return {
            "neighbor_indices": valid_indices,
            "mask": mask,
            "distances": d_valid
        }

    def _compute_sdf(self, mesh, query_points):
        """Computes Signed Distance Function using trimesh."""
        # query_points shape: (nx, ny, nz, 3)
        flat_queries = query_points.reshape(-1, 3)
        
        # trimesh signed_distance returns (n_queries,)
        import trimesh
        try:
            # This requires the mesh to be watertight for a "true" signed distance
            # For Car-CFD, we often use unsigned distance as a safe proxy if not watertight
            if mesh.is_watertight:
                sdf = trimesh.proximity.signed_distance(mesh, flat_queries)
            else:
                # Use unsigned distance if not watertight
                sdf = trimesh.proximity.ProximityQuery(mesh).vertex(flat_queries)[0]
                # Alternatively, use absolute distance to surface (slower but more accurate)
                # _, sdf, _ = trimesh.proximity.closest_point(mesh, flat_queries)
        except Exception as e:
            print(f"SDF Error: {e}")
            # Fallback to nearest-vertex distance
            tree = scipy.spatial.KDTree(mesh.vertices)
            sdf, _ = tree.query(flat_queries)
            
        return sdf.reshape((*self.query_res, 1)).astype(np.float32)

    # --------------------------------------------------
    def _pad(self, arr):
        n = arr.shape[0]
        if n >= self.max_vertices:
            return arr[:self.max_vertices]

        pad = np.zeros((self.max_vertices - n, arr.shape[1]))
        return np.vstack([arr, pad])

    # --------------------------------------------------
    def _generate_grid(self):
        x = np.linspace(0, 1, self.query_res[0])
        y = np.linspace(0, 1, self.query_res[1])
        z = np.linspace(0, 1, self.query_res[2])

        grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        return grid.astype(np.float32)

    # --------------------------------------------------
    def get_batch(self, split="train", batch_size=1):
        data = self.train_data if split == "train" else self.test_data

        idxs = np.arange(len(data))
        np.random.shuffle(idxs)

        for i in range(0, len(idxs), batch_size):
            batch_idx = idxs[i:i+batch_size]

            batch = {}
            # We assume all samples have the same keys
            for k in data[0].keys():
                val0 = data[batch_idx[0]][k]
                
                if isinstance(val0, dict):
                    # Handle nested neighbor dicts
                    batch[k] = {}
                    for sub_k in val0.keys():
                        batch[k][sub_k] = np.stack([data[j][k][sub_k] for j in batch_idx], axis=0)
                else:
                    batch[k] = np.stack([data[j][k] for j in batch_idx], axis=0)

            yield jax.tree_util.tree_map(lambda x: jnp.array(x) if isinstance(x, (np.ndarray, list)) else x, batch)
def load_mini_car():
    """
    JAX-compatible mini dataset loader
    """
    path = Path("data/mini_car.npz")

    if not path.exists():
        raise FileNotFoundError("mini_car.npz not found")

    data = np.load(path, allow_pickle=True)
    return {k: jnp.array(data[k]) for k in data.files}