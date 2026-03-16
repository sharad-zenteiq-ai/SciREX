import json
import urllib.request
from typing import List, Union, Dict, Iterator
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
        self.n_train = n_train
        self.n_test = n_test
        self.query_res = query_res

        self.item_dir_name = "processed-car-pressure-data"
        self.data_dir = self.root_dir / self.item_dir_name

        if download and not self.data_dir.exists():
            download_from_zenodo_record(record_id=self.zenodo_record_id, root=self.root_dir)

        # Load samples assuming the extracted data format is compatible (.npz mappings)
        self.train_data = self._load_data(split="train", n_samples=n_train)
        self.test_data = self._load_data(split="test", n_samples=n_test)

        # Clean/preprocess instances natively using numpy
        self._process_data(self.train_data)
        self._process_data(self.test_data)

    def _load_data(self, split: str, n_samples: int) -> List[Dict[str, np.ndarray]]:
        """
        Loads dataset examples dynamically looking for matching `.npz` objects.
        """
        if not self.data_dir.exists():
            print(f"Warning: directory '{self.data_dir}' does not exist. Returning empty `{split}` dataset.")
            return []

        data_list = []
        files = sorted(list(self.data_dir.glob(f"{split}_*.npz")))
        
        # Fallback to general files if specific split naming isn't maintained
        if not files:
            files = sorted(list(self.data_dir.glob("*.npz")))

        for i, file_path in enumerate(files[:n_samples]):
            data_dict = np.load(file_path, allow_pickle=True)
            # Unpack NpzFile into native dictionary of numpy arrays
            ready_dict = {k: data_dict[k] for k in data_dict.files}
            data_list.append(ready_dict)
            
        if len(data_list) < n_samples:
            print(f"Warning: Requested {n_samples} {split} samples, but only {len(data_list)} available files were found.")

        return data_list

    def _process_data(self, data_list: List[Dict[str, np.ndarray]]):
        """
        Processes the data list in-place to format instances matching the mesh definitions.
        """
        for i, data in enumerate(data_list):
            if "press" in data:
                press = data["press"]
                # NumPy replacement of `torch.cat((press[:, 0:16], press[:, 112:]), axis=1)`
                data_list[i]["press"] = np.concatenate(
                    (press[:, 0:16], press[:, 112:]), axis=1
                )

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
