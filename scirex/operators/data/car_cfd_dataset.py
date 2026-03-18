from typing import List, Union, Dict, Optional
from pathlib import Path
import json
import urllib.request
import numpy as np
import jax.numpy as jnp


# --------------------------------------------------
# 🔽 Download utility (no neuralop)
# --------------------------------------------------
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


# --------------------------------------------------
# 🔽 JAX Dataset (NO MeshDataModule)
# --------------------------------------------------
class CarCFDDataset:
    def __init__(
        self,
        root_dir: Union[str, Path],
        n_train: int = 1,
        n_test: int = 1,
        query_res: List[int] = [32, 32, 32],
        download: bool = True,
        max_vertices: int = 4096,   # ⚠️ IMPORTANT for JAX
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.query_res = query_res
        self.max_vertices = max_vertices
        self.zenodo_record_id = "13936501"

        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True)

        if download and not (self.root_dir / "processed-car-pressure-data").exists():
            download_from_zenodo_record(self.zenodo_record_id, self.root_dir)

        self.base_dir, self.data_dir = self._find_data_dir()

        self.train_data = self._load_split("train.txt", n_train)
        self.test_data = self._load_split("test.txt", n_test)

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
    def _load_split(self, split_file, n_samples):
        path = self.base_dir / split_file

        with open(path) as f:
            indices = [i.strip() for i in f.readline().split(",") if i.strip()]

        data = []
        for idx in indices[:n_samples]:
            item = self._load_item(idx)
            if item:
                data.append(item)

        return data

    # --------------------------------------------------
    def _load_item(self, idx: str) -> Optional[Dict]:
        npz_path = self.data_dir / f"{idx}.npz"

        if not npz_path.exists():
            return None

        with np.load(npz_path, allow_pickle=True) as data:
            item = {k: data[k] for k in data.files}

        vertices = item["vertices"]
        press = item["press"]

        # 🔥 EXACT same logic as torch version
        if press.ndim == 2 and press.shape[1] > 112:
            press = np.concatenate((press[:, 0:16], press[:, 112:]), axis=1)

        if press.ndim == 1:
            press = press[:, None]

        # 🔴 CRITICAL: Fix size for JAX
        vertices = self._pad(vertices)
        press = self._pad(press)

        query_points = self._generate_grid()
        distance = np.zeros((*self.query_res, 1), dtype=np.float32)

        return {
            "vertices": vertices.astype(np.float32),
            "press": press.astype(np.float32),
            "query_points": query_points,
            "distance": distance,
        }

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
            for k in data[0].keys():
                batch[k] = np.stack([data[j][k] for j in batch_idx], axis=0)

            # 🔥 convert to JAX
            yield {k: jnp.array(v) for k, v in batch.items()}
def load_mini_car():
    """
    JAX-compatible mini dataset loader
    """
    path = Path("data/mini_car.npz")

    if not path.exists():
        raise FileNotFoundError("mini_car.npz not found")

    data = np.load(path, allow_pickle=True)
    return {k: jnp.array(data[k]) for k in data.files}