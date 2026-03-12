### NSDataset Utility

The **NSDataset** class serves as the core data handling component for Navier-Stokes super-resolution tasks. It streamlines the pipeline through the following mechanisms:

* It loads pre-processed **Low-Resolution (LR)** and **High-Resolution (HR)** data stored as NumPy arrays.
* The utility structures the data into individual **per-time-step samples**, ensuring the model receives appropriate temporal snapshots for training.
* It prepares the arrays for the **Bicubic FNO** architecture by ensuring spatial and temporal consistency between the interpolated baseline and the ground truth.

---

### Implementation Details

| Feature | Description |
| :--- | :--- |
| **Data Format** | NumPy arrays (.npy) |
| **Input Pairs** | Coarse-grid inputs and corresponding fine-grid labels |
| **Output Type** | PyTorch-compatible per-time-step tensors |

