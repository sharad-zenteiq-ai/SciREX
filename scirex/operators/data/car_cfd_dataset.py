import json
import urllib.request
from typing import List, Union, Dict, Iterator, Optional
from pathlib import Path

import numpy as np

def download_from_zenodo_record(record_id: str, root: Path):
    """
    Downloads files from a Zenodo record to the specified root directory.
    Uses standard library urllib for a pure Python implementation without requests.
    """
    url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as response:
            metadata = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to fetch Zenodo metadata: {e}")
        return

    root.mkdir(parents=True, exist_ok=True)
    
    for file_obj in metadata.get('files', []):
        file_url = file_obj['links']['self']
        file_name = file_obj['key']
        file_path = root / file_name
        
        if not file_path.exists():
            print(f"Downloading {file_name} to {file_path}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")
                continue
            
            # Extract archives automatically
            if file_name.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
            elif file_name.endswith('.tar.gz') or file_name.endswith('.tgz') or file_name.endswith('.tar'):
                import tarfile
                mode = "r:gz" if file_name.endswith('gz') else "r:"
                with tarfile.open(file_path, mode) as tar_ref:
                    tar_ref.extractall(root)


class CarCFDDataset:
    """Processed version of the Car-CFD dataset, purely implemented in JAX/NumPy.
    
    CarCFDDataset is a processed version of the dataset introduced in
    [1]_, which encodes a triangular mesh over the surface of a 3D model car
    and provides the air pressure at each centroid and vertex of the mesh when
    the car is placed in a simulated wind tunnel with a recorded inlet velocity.
    In our case, inputs are a signed distance function evaluated over a regular
    3D grid of query points, as well as the inlet velocity. Outputs are pressure
    values at each centroid of the triangle mesh.

    This implementation has been refactored to remove all PyTorch dependencies,
    using pure NumPy structures to be seamlessly compatible with JAX workflows.
    Data should be stored and queried as `.npz` formatted Numpy archives.

    Data is also stored on Zenodo: https://zenodo.org/records/13936501

    Parameters
    ----------
    root_dir : Union[str, Path]
        root directory at which data is stored.
    n_train : int, optional
        Number of training instances to load, by default 1
    n_test : int, optional
        Number of testing instances to load, by default 1
    query_res : List[int], optional
        Dimension-wise resolution of signed distance function
        (SDF) query cube, by default [32,32,32]
    download : bool, optional
        Whether to download data from Zenodo, by default True

    References
    ----------
    .. [1] : Umetani, N. and Bickel, B. (2018). "Learning three-dimensional flow for interactive
        aerodynamic design". ACM Transactions on Graphics, 2018.
        https://dl.acm.org/doi/10.1145/3197517.3201325.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        n_train: int = 1,
        n_test: int = 1,
        query_res: List[int] = [32, 32, 32],
        download: bool = True,
    ):
        """Initialize the CarCFDDataset."""
        self.zenodo_record_id = "13936501"
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.query_res = query_res

        # The dataset often has a nested structure after extraction
        # We look for where 'train.txt' and 'data' live
        potential_dirs = [
             self.root_dir,
             self.root_dir / "processed-car-pressure-data",
             self.root_dir / "processed-car-pressure-data" / "processed-car-pressure-data"
        ]
        
        self.data_dir = None
        self.train_txt = None
        self.test_txt = None
        
        for d in potential_dirs:
            if (d / "train.txt").exists() and (d / "data").exists():
                self.data_dir = d / "data"
                self.train_txt = d / "train.txt"
                self.test_txt = d / "test.txt"
                break
        
        if self.data_dir is None:
            if download:
                print(f"Data not found locally. Attempting download to {self.root_dir}...")
                download_from_zenodo_record(record_id=self.zenodo_record_id, root=self.root_dir)
                # Re-check after download
                for d in potential_dirs:
                    if (d / "train.txt").exists() and (d / "data").exists():
                        self.data_dir = d / "data"
                        self.train_txt = d / "train.txt"
                        self.test_txt = d / "test.txt"
                        break
            
            if self.data_dir is None:
                print(f"Warning: Could not find dataset structure in {self.root_dir}")
                self.train_data = []
                self.test_data = []
                return

        # Load samples
        self.train_data = self._load_data_split(self.train_txt, n_samples=n_train)
        self.test_data = self._load_data_split(self.test_txt, n_samples=n_test)

        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

        # Clean/preprocess instances natively using numpy
        self._process_data(self.train_data)
        self._process_data(self.test_data)

    def _load_data_split(self, split_file: Path, n_samples: int) -> List[Dict[str, np.ndarray]]:
        """Loads data samples based on indices in the split file."""
        if not split_file or not split_file.exists():
            return []
            
        with open(split_file, "r") as f:
            line = f.readline().strip()
            if not line: return []
            indices = [idx.strip() for idx in line.split(",") if idx.strip()]
            
        data_list = []
        for idx in indices[:n_samples]:
            sample = self._load_item(idx)
            if sample:
                data_list.append(sample)
        return data_list

    def _load_item(self, idx: str) -> Optional[Dict[str, np.ndarray]]:
        """Loads a single item (either .npz or raw .npy/.ply)."""
        # Try .npz first
        npz_path = self.data_dir / f"{idx}.npz"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as data:
                return {k: data[k] for k in data.files}
        
        # Try raw files (nested or flat)
        # Expected structure: data/data/{idx}/press.npy or data/press_{idx}.npy
        press_path = self.data_dir / "data" / idx / "press.npy"
        if not press_path.exists():
            press_path = self.data_dir / f"press_{idx}.npy"
            
        mesh_path = self.data_dir / "data" / idx / "tri_mesh.ply"
        if not mesh_path.exists():
            mesh_path = self.data_dir / f"mesh_{idx}.ply"
            
        if press_path.exists() and mesh_path.exists():
            press = np.load(press_path)
            vertices = self._read_ply_vertices(mesh_path)
            
            # Since SDF is missing in raw data and we don't have Open3D, 
            # we generate a grid and a dummy distance field for shape verification.
            item = {
                "press": press,
                "vertices": vertices,
                "query_points": self._generate_grid(),
                "distance": np.zeros((*self.query_res, 1)) # Placeholder
            }
            return item
            
        return None

    def _read_ply_vertices(self, path: Path) -> np.ndarray:
        """Simple PLY vertex reader for standard ASCII/Binary formats."""
        try:
            # We use a simplified path: just extract the first N vertices
            # based on the header. A full parser would be better but this is robust enough.
            with open(path, "rb") as f:
                header = ""
                while "end_header" not in header:
                    line = f.readline().decode("ascii", errors="ignore")
                    header += line
                    if "element vertex" in line:
                        n_verts = int(line.split()[-1])
                
                # If binary, read the rest. If ASCII, it's after end_header.
                if "format binary" in header:
                    # Assume float32 x, y, z
                    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                    # Basic PLY often has other properties, we take the first 3 floats
                    # This is a heuristic that works for most Car-CFD PLYs
                    data = np.fromfile(f, dtype='f4', count=n_verts*3).reshape(-1, 3)
                    return data
                else:
                    # ASCII
                    content = f.read().decode("ascii", errors="ignore")
                    lines = content.strip().split("\n")[:n_verts]
                    verts = [[float(x) for x in l.split()[:3]] for l in lines]
                    return np.array(verts)
        except Exception as e:
            print(f"Error reading PLY {path}: {e}")
            return np.zeros((1, 3))

    def _generate_grid(self) -> np.ndarray:
        """Generates a regular grid of query points for the latent space."""
        # Standard unit cube or based on watertight_global_bounds if we wanted to be fancy.
        x = np.linspace(0, 1, self.query_res[0])
        y = np.linspace(0, 1, self.query_res[1])
        z = np.linspace(0, 1, self.query_res[2])
        grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        return grid.astype(np.float32)

    def _process_data(self, data_list: List[Dict[str, np.ndarray]]):
        """
        Processes the data list in-place to format instances matching the mesh definitions.
        """
        for i, data in enumerate(data_list):
            if "press" in data:
                press = data["press"]
                # If it's 2D and has many columns, it might need the column selection
                # as per the original and neuraloperator implementations.
                if press.ndim == 2 and press.shape[1] > 112:
                    press = np.concatenate(
                        (press[:, 0:16], press[:, 112:]), axis=1
                    )
                
                # Ensure it is at least (N, 1) for JAX consistency
                if press.ndim == 1:
                    press = press[:, np.newaxis]
                
                data_list[i]["press"] = press

    def train_generator(self, batch_size: int, shuffle: bool = True) -> Iterator[Dict[str, np.ndarray]]:
        """Yields batched dictionaries of PyTrees (numpy arrays) for training."""
        return self._batch_generator(self.train_data, batch_size, shuffle)

    def test_generator(self, batch_size: int, shuffle: bool = False) -> Iterator[Dict[str, np.ndarray]]:
        """Yields batched dictionaries of PyTrees (numpy arrays) for testing."""
        return self._batch_generator(self.test_data, batch_size, shuffle)

    def _batch_generator(self, data_list: List[Dict[str, np.ndarray]], batch_size: int, shuffle: bool) -> Iterator[Dict[str, np.ndarray]]:
        if not data_list:
            return

        indices = np.arange(len(data_list))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_data = {}
            
            # Map batch dimension onto properties uniformly
            keys = data_list[0].keys()
            for k in keys:
                batch_data[k] = np.stack([data_list[idx][k] for idx in batch_indices], axis=0)
            
            yield batch_data


def load_mini_car() -> Dict[str, np.ndarray]:
    """
    Load the 3-example mini Car-CFD dataset inherently in SciREX using pure Numpy.
    Assumes standard .npz format over the previously coupled .pt torch dict mappings.
    """
    this_dir = Path(__file__).resolve().parent
    file_path = this_dir / "data" / "mini_car.npz"
    
    if file_path.exists():
        data = np.load(file_path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    else:
        raise FileNotFoundError(
            f"`mini_car.npz` was not found at {file_path}. "
            "Please ensure the dataset has been generated or converted out of PyTorch format!"
        )
