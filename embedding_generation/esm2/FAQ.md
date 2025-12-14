## Do I need to install fair-esm?

No! The ESM library is included in the `esm/` folder in this repository. You only need to install:
- PyTorch
- NumPy

Just make sure you run the script from the current directory.

## Where are the ESM2 model weights stored?

The first time you use a specific ESM2 model, the script will download the model weights from the internet. These are typically cached in:
- `~/.cache/torch/hub/checkpoints/` (Linux/Mac)
- `C:\Users\YourName\.cache\torch\hub\checkpoints\` (Windows)

The model weights are reused for future runs, so you only download them once.
