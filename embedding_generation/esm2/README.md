# ESM2 Embedding Generator

Generate ESM2 protein language model embeddings for protein sequences. This tool processes FASTA files and produces embeddings that can be used for downstream tasks like DNA-binding prediction with PRIMED.

## Overview

This script extracts per-residue embeddings from ESM2 (Evolutionary Scale Modeling) protein language models. ESM2 models are trained on millions of protein sequences and capture evolutionary and structural information.

**Output**: 2560-dimensional embeddings per residue (for `esm2_t36_3B_UR50D`)

## Requirements
```bash
pip install torch numpy
```

- Python >= 3.7
- PyTorch >= 1.9
- NumPy >= 1.19
- ESM library (included in `esm/` folder)

**Note**: ESM2 models are large and require significant GPU memory:
- `esm2_t6_8M_UR50D`: ~35 MB (smallest)
- `esm2_t12_35M_UR50D`: ~150 MB
- `esm2_t30_150M_UR50D`: ~600 MB
- `esm2_t33_650M_UR50D`: ~2.5 GB
- `esm2_t36_3B_UR50D`: ~11 GB (default)
- `esm2_t48_15B_UR50D`: ~55 GB (largest)

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/esm2-embeddings.git
cd esm2-embeddings

# Install dependencies
pip install -r requirements.txt

# The ESM library is included in the esm/ folder
# No additional installation needed
```

## Repository Structure
```
esm2-embeddings/
├── generate_esm2_embeddings.py   # Main script
├── esm/                          # ESM library (included)
│   ├── pretrained.py
│   ├── data.py
│   └── ...
├── README.md
├── requirements.txt
├── LICENSE
└── example_data/
    └── proteins.fasta
```

## Input Format

### Supported FASTA Formats

**1. Standard FASTA (2 lines per entry):**
```
>protein1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK
>protein2
MKKLVLSLSLVLAFSSATAAF
```

**2. Extended FASTA with labels (3 lines per entry):**
```
>protein1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK
0000000000111111000000000111100000000000000000000000000000000
>protein2
MKKLVLSLSLVLAFSSATAAF
0000000000000000000000
```

- Line 1: Header (starts with `>`)
- Line 2: Protein sequence (standard amino acids)
- Line 3: Binary labels (`0` = non-binding, `1` = binding) - **optional**

## Usage

### Basic Usage
```bash
python generate_esm2_embeddings.py \
  --input-files proteins.fasta \
  --output-dir ./esm2_embeddings
```

### Multiple Input Files
```bash
python generate_esm2_embeddings.py \
  --input-files file1.fasta file2.fasta file3.fasta \
  --output-dir ./embeddings/esm2
```

### Different Model Sizes
```bash
# Use smaller model (faster, less memory)
python generate_esm2_embeddings.py \
  --input-files proteins.fasta \
  --model-name esm2_t33_650M_UR50D \
  --output-dir ./esm2_embeddings

# Use largest model (best quality, requires ~55GB GPU memory)
python generate_esm2_embeddings.py \
  --input-files proteins.fasta \
  --model-name esm2_t48_15B_UR50D \
  --output-dir ./esm2_embeddings
```

### With All Options
```bash
python generate_esm2_embeddings.py \
  --input-files DNA-46_Test.txt DNA-129_Test.txt \
  --output-dir ./embeddings/esm2 \
  --output-prefix esm2_ \
  --model-name esm2_t36_3B_UR50D \
  --max-tokens 1022 \
  --normalize \
  --device cuda \
  --verbose
```

### CPU Mode (No GPU)
```bash
python generate_esm2_embeddings.py \
  --input-files proteins.fasta \
  --output-dir ./esm2_embeddings \
  --device cpu
```

## Command-Line Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--input-files` | FASTA file(s) to process (space-separated for multiple) |

### Optional

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `./esm2_embeddings` | Directory to save embeddings |
| `--model-name` | `esm2_t36_3B_UR50D` | ESM2 model variant (see options below) |
| `--max-tokens` | 1022 | Maximum tokens per chunk |
| `--output-prefix` | `""` | Prefix for output filenames |
| `--normalize` | `True` | Apply z-score normalization |
| `--no-normalize` | - | Disable normalization |
| `--save-labels` | `True` | Save labels if present in FASTA |
| `--device` | `auto` | Device: `auto`, `cuda`, or `cpu` |
| `--verbose` | `False` | Print detailed processing info |

### Available ESM2 Models

| Model Name | Parameters | GPU Memory | Embedding Dim |
|------------|-----------|------------|---------------|
| `esm2_t6_8M_UR50D` | 8M | ~0.5 GB | 320 |
| `esm2_t12_35M_UR50D` | 35M | ~1 GB | 480 |
| `esm2_t30_150M_UR50D` | 150M | ~2 GB | 640 |
| `esm2_t33_650M_UR50D` | 650M | ~4 GB | 1280 |
| `esm2_t36_3B_UR50D` | 3B | ~12 GB | **2560** (default) |
| `esm2_t48_15B_UR50D` | 15B | ~60 GB | 5120 |

**Note**: The default model (`esm2_t36_3B_UR50D`) produces 2560-dimensional embeddings, which is required for PRIMED inference.

## Output Format

### Output Files

Each protein is saved as: `{output_prefix}{protein_name}.pt`

Example:
```
esm2_embeddings/
├── protein1.pt
├── protein2.pt
└── protein3.pt
```

### File Structure
```python
torch.load("protein1.pt")
# Returns:
{
    "embedding": torch.Tensor,  # Shape: [seq_length, 2560]
    "header": str,              # Protein identifier
    "label": torch.Tensor       # Shape: [seq_length, 1] (if labels present)
}
```

**Example:**
```python
import torch

data = torch.load("protein1.pt")
print(f"Protein: {data['header']}")
print(f"Embedding shape: {data['embedding'].shape}")  # [150, 2560] for 150 residues
if 'label' in data:
    print(f"Label shape: {data['label'].shape}")      # [150, 1]
```

## Features

### Automatic Sequence Chunking

Long sequences (>1022 residues) are automatically chunked and then concatenated:
```
Input:  2500 residue protein
        ↓
Chunks: [1022] + [1022] + [456]
        ↓
Process each chunk through ESM2
        ↓
Output: [2500, 2560] concatenated embedding
```

### Z-Score Normalization

When `--normalize` is enabled (default):
```python
embedding = (embedding - mean) / (std + 1e-6)
```

This ensures consistent scale across proteins and improves downstream model performance.

### Memory Management

- Processes one protein at a time
- Automatic GPU cache clearing after each protein
- Efficient for large datasets

## Example Workflows

### Workflow 1: Generate Embeddings for PRIMED
```bash
# Generate ESM2 embeddings
python generate_esm2_embeddings.py \
  --input-files test_proteins.fasta \
  --output-dir ./embeddings/esm2 \
  --model-name esm2_t36_3B_UR50D \
  --verbose

# Use with PRIMED (requires ESM3 and ESMC embeddings too)
python primed_inference.py \
  --esm2-dir ./embeddings/esm2 \
  --esm3-dir ./embeddings/esm3 \
  --esmc-dir ./embeddings/esmc \
  --test-datasets test_proteins.fasta \
  --model-path primed_model.pth
```

### Workflow 2: Batch Processing Multiple Datasets
```bash
# Generate embeddings for all FASTA files
python generate_esm2_embeddings.py \
  --input-files dataset1.fasta dataset2.fasta dataset3.fasta \
  --output-dir ./all_embeddings/esm2 \
  --output-prefix exp1_ \
  --verbose
```

### Workflow 3: Use Smaller Model (Limited GPU Memory)
```bash
# Use 650M model instead of 3B
python generate_esm2_embeddings.py \
  --input-files proteins.fasta \
  --model-name esm2_t33_650M_UR50D \
  --output-dir ./esm2_650M \
  --device cuda
```

**Note**: Smaller models produce different embedding dimensions, which may not be compatible with pre-trained PRIMED models.

## Performance Tips

1. **GPU Acceleration**: Use CUDA for 10-100x speedup
```bash
   --device cuda
```

2. **Batch Processing**: Process multiple files in one command
```bash
   --input-files *.fasta
```

3. **Model Selection**: 
   - For speed: `esm2_t33_650M_UR50D`
   - For quality: `esm2_t36_3B_UR50D` (default)
   - For best results: `esm2_t48_15B_UR50D` (requires 60GB GPU)

4. **Monitor Memory**: Use `nvidia-smi` to check GPU usage

## Troubleshooting

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use a smaller model:
```bash
   --model-name esm2_t33_650M_UR50D
```

2. Use CPU mode:
```bash
   --device cpu
```

3. Reduce `--max-tokens`:
```bash
   --max-tokens 512
```

### Import Error: esm module

**Error:**
```
ModuleNotFoundError: No module named 'esm'
```

**Solution:**
Make sure you're running the script from the repository root directory where the `esm/` folder is located:
```bash
cd /path/to/esm2-embeddings
python generate_esm2_embeddings.py --input-files proteins.fasta
```

### Model Download Issues

The first time you run the script with a specific model, it will download the model weights from the internet. This requires:
- Internet connection
- Sufficient disk space (~11GB for default model)
- The weights are cached for future use

### Empty Output Directory

**Possible causes:**
- Invalid FASTA format
- All sequences skipped due to errors
- Check with `--verbose` flag

### Slow Processing on CPU

ESM2 models are computationally intensive. CPU processing is 10-100x slower than GPU.

**Estimated times (esm2_t36_3B_UR50D, 500 residues):**
- GPU (A100): ~2 seconds
- GPU (V100): ~5 seconds
- CPU: ~2-5 minutes

## Technical Details

### Model Architecture

ESM2 uses a transformer encoder architecture similar to BERT:
- Input: Amino acid sequence
- Output: Contextualized embeddings per residue
- Training: Masked language modeling on 250M protein sequences

### Embedding Extraction

Embeddings are extracted from the **final layer** of the transformer:
```python
representations = model(tokens, repr_layers=[model.num_layers])
embeddings = representations["representations"][model.num_layers]
```

### Normalization Details

Per-residue z-score normalization:
```python
mean = embeddings.mean(dim=1, keepdim=True)  # Mean across embedding dimension
std = embeddings.std(dim=1, keepdim=True)    # Std across embedding dimension
normalized = (embeddings - mean) / (std + 1e-6)
```

## Citation

If you use ESM2 embeddings, please cite:
```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yilun and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```

## Related Tools

- **ESM3 Embeddings**: Generate ESM3 embeddings (1536-dim)
- **ESMC Embeddings**: Generate ESMC embeddings (1152-dim)
- **PRIMED**: DNA-binding prediction using ESM2+ESM3+ESMC embeddings

## License

This project is licensed under the MIT License.

## Contact

For questions or issues:
- Contact Luoshu Zhang at luoshu.zhang@ufl.edu for further details.
