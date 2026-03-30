import os
import json
import urllib.request
from pathlib import Path

import os
import json
import urllib.request
from pathlib import Path


def download_dataset(root: Path):
    record_id = "13936501"
    url = f"https://zenodo.org/api/records/{record_id}"

    root.mkdir(parents=True, exist_ok=True)

    print("Fetching metadata...")
    with urllib.request.urlopen(url) as response:
        metadata = json.loads(response.read().decode())

    for file_obj in metadata.get("files", []):
        file_url = file_obj["links"]["self"]
        file_name = file_obj["key"]
        file_path = root / file_name

        expected_size = file_obj.get("size", None)

        # --------------------------------------------------
        # CHECK CORRUPTION
        # --------------------------------------------------
        if file_path.exists() and expected_size:
            actual_size = file_path.stat().st_size

            if actual_size != expected_size:
                print(f" Corrupted file detected: {file_name}")
                print("Deleting and re-downloading...")
                file_path.unlink()

        # --------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------
        if not file_path.exists():
            print(f"⬇ Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)

        # --------------------------------------------------
        # EXTRACT (SAFE)
        # --------------------------------------------------
        print(f" Extracting {file_name}...")

        try:
            if file_name.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(file_path, "r") as z:
                    z.extractall(root)

            elif file_name.endswith((".tar.gz", ".tgz", ".tar")):
                import tarfile
                mode = "r:gz" if file_name.endswith("gz") else "r:"
                with tarfile.open(file_path, mode) as t:
                    t.extractall(root)

        except Exception as e:
            print(f" Extraction failed: {e}")
            print("Deleting corrupted file and retrying...\n")

            file_path.unlink()

            # Retry download once
            print(f" Re-downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)

            print(f" Extracting again...")
            import tarfile
            mode = "r:gz" if file_name.endswith("gz") else "r:"
            with tarfile.open(file_path, mode) as t:
                t.extractall(root)
                
# --------------------------------------------------
# ROBUST DATA DIR FINDER
# --------------------------------------------------
def find_data_dir(root):

    for path in root.rglob("*"):
        if path.name == "data" and path.is_dir():
            print(f" Found data dir: {path}")
            return path

    print("\n Could not find data folder. Available dirs:")
    for p in root.rglob("*"):
        if p.is_dir():
            print(p)

    raise RuntimeError("Data directory not found")


# --------------------------------------------------
# SAVE PLY
# --------------------------------------------------
def save_ply(vertices, faces, filepath):
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


# --------------------------------------------------
# CONVERT NPZ → SAMPLE FOLDERS
# --------------------------------------------------
def convert_to_sample_folders(data_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*.npz"))

    print(f"\n Total NPZ files found: {len(npz_files)}")

    for i, npz_file in enumerate(npz_files):
        idx = npz_file.stem
        sample_dir = out_dir / idx
        sample_dir.mkdir(exist_ok=True)

        data = np.load(npz_file, allow_pickle=True)

        vertices = data["vertices"]
        press = data["press"]

        # SAME FIX AS ORIGINAL DATASET
        if press.ndim == 2 and press.shape[1] > 112:
            press = np.concatenate((press[:, 0:16], press[:, 112:]), axis=1)

        # Save pressure
        np.save(sample_dir / "press.npy", press)

        # Save mesh
        if "faces" in data:
            faces = data["faces"]
            save_ply(vertices, faces, sample_dir / "mesh.ply")
        else:
            print(f" No faces in {idx}, skipping mesh")

        if i % 50 == 0:
            print(f"Processed {i}/{len(npz_files)}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    root = Path("folder/data/sample")

    # Step 1: Download + Extract
    download_dataset(root)

    # Step 2: Find actual data folder
    data_dir = find_data_dir(root)

    # Step 3: Convert into sample-wise folders
    out_dir = root / "samples"
    convert_to_sample_folders(data_dir, out_dir)
