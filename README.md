# PRIMED Inference

PRIMED (Protein Residue Inference using MLP for Enhanced DNA-binding predictions) is a tool for predicting DNA-binding residues in proteins using concatenated embeddings from ESM2, ESM3, and ESMC protein language models.

## Requirements
```bash
pip install torch numpy biopython scikit-learn
```

- Python >= 3.7
- PyTorch >= 1.9
- NumPy >= 1.19
- BioPython >= 1.78
- scikit-learn >= 0.24

## Quick Start with Example Data

We provide example data to help you get started:
```bash
python primed_inference.py \
  --esm2-dir ./embedding_examples/esm2 \
  --esm3-dir ./embedding_examples/esm3 \
  --esmc-dir ./embedding_examples/esmc \
  --test-datasets DNA-46_Test.txt \
  --model-path primed_model.pth
```

**Example files included:**
- `DNA-46_Test.txt` - Sample test dataset with 46 proteins
- `embedding_examples/esm2/` - Example ESM2 embeddings
- `embedding_examples/esm3/` - Example ESM3 embeddings
- `embedding_examples/esmc/` - Example ESMC embeddings

## Generating Your Own Embeddings

If you want to generate embeddings for your own protein sequences, please refer to the embedding generation scripts in the `embedding_generation/` directory. This includes:

- Scripts for generating ESM2 embeddings (2560-dim)
- Scripts for generating ESM3 embeddings (1536-dim)
- Scripts for generating ESMC embeddings (1152-dim)
- Instructions for processing custom FASTA files

See `embedding_generation/README.md` for detailed instructions.

## Input Requirements

### 1. Protein Language Model Embeddings

Three directories containing embeddings for each protein:
```
embeddings/
├── esm2/          # ESM2 embeddings (2560-dim)
│   ├── protein1.pt
│   └── protein2.pt
├── esm3/          # ESM3 embeddings (1536-dim)
│   ├── protein1.pt
│   └── protein2.pt
└── esmc/          # ESMC embeddings (1152-dim)
    ├── protein1.pt
    └── protein2.pt
```

**Embedding file format:**
- ESM2/ESM3: `{"embedding": torch.Tensor}` with shape `[seq_length, dim]`
- ESMC: `{"emb": torch.Tensor}` with shape `[seq_length, 1152]`
- Protein names must match exactly across all three directories

### 2. Test Dataset (FASTA format)

Three-line format per protein:
```
>protein1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK
0000000000111111000000000111100000000000000000000000000000000
>protein2
MKKLVLSLSLVLAFSSATAAF
0000000000000000000000
```

- Line 1: Header (starts with `>`)
- Line 2: Protein sequence
- Line 3: Binary labels (`0` = non-binding, `1` = binding)

**Note**: Label length must match sequence length exactly.

### 3. Trained Model

- PyTorch model file (`.pth`)
- Optional: Threshold file (default: `optimized_threshold.txt`)

## Usage

### Basic Usage
```bash
python primed_inference.py \
  --esm2-dir ./embeddings/esm2 \
  --esm3-dir ./embeddings/esm3 \
  --esmc-dir ./embeddings/esmc \
  --test-datasets test.fasta \
  --model-path primed_model.pth
```

### Multiple Datasets
```bash
python primed_inference.py \
  --esm2-dir ./embeddings/esm2 \
  --esm3-dir ./embeddings/esm3 \
  --esmc-dir ./embeddings/esmc \
  --test-datasets dataset1.fasta dataset2.fasta dataset3.fasta \
  --dataset-names "Test-A" "Test-B" "Test-C" \
  --model-path primed_model.pth
```

### Save Prediction Scores
```bash
python primed_inference.py \
  --esm2-dir ./embeddings/esm2 \
  --esm3-dir ./embeddings/esm3 \
  --esmc-dir ./embeddings/esmc \
  --test-datasets test.fasta \
  --model-path primed_model.pth \
  --save-scores \
  --verbose
```

### Custom Configuration
```bash
python primed_inference.py \
  --esm2-dir ./embeddings/esm2 \
  --esm3-dir ./embeddings/esm3 \
  --esmc-dir ./embeddings/esmc \
  --test-datasets test.fasta \
  --model-path primed_model.pth \
  --threshold 0.813 \
  --output-dir ./results \
  --output-prefix evaluation \
  --save-scores
```

## Command-Line Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--esm2-dir` | Directory containing ESM2 embeddings (.pt files) |
| `--esm3-dir` | Directory containing ESM3 embeddings (.pt files) |
| `--esmc-dir` | Directory containing ESMC embeddings (.pt files) |
| `--test-datasets` | FASTA file(s) (space-separated for multiple) |
| `--model-path` | Path to trained model (.pth file) |

### Optional

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-names` | Auto-generated | Custom names for datasets |
| `--input-dim` | 5248 | Input dimension (2560+1536+1152) |
| `--threshold` | From file | Classification threshold |
| `--threshold-file` | `optimized_threshold.txt` | File containing threshold |
| `--output-dir` | `./results` | Output directory |
| `--output-prefix` | `primed` | Prefix for output filenames |
| `--save-scores` | False | Save raw prediction scores to JSON |
| `--device` | `auto` | Device: `auto`, `cuda`, or `cpu` |
| `--verbose` | False | Print detailed processing info |

## Output Files

### Metrics Table (always generated)

**File**: `{output_prefix}_metrics.txt`

Tab-delimited performance metrics:
```
Dataset    AUC     MCC     Sensitivity  Specificity  Threshold  Proteins  Residues  Binding_Sites  Binding_%
TestSet    0.8523  0.6234  0.7891      0.8234       0.8130     150       45231     5623          12.43
```

### Prediction Scores (optional, with `--save-scores`)

**File**: `{output_prefix}_scores_{dataset_name}.json`

Raw prediction probabilities:
```json
{
  "protein1": [0.123, 0.456, 0.789, ...],
  "protein2": [0.345, 0.678, 0.901, ...],
  ...
}
```

## Model Architecture
```
Input: Concatenated embeddings [ESM2(2560) + ESM3(1536) + ESMC(1152)] = 5248
  ↓
LayerNorm(5248) → Linear(5248→256) → ReLU
  ↓
Linear(256→64) → ReLU
  ↓
Linear(64→1) → Sigmoid
```

## Project Structure
```
primed/
├── primed_inference.py           # Main inference script
├── README.md                     # This file
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── optimized_threshold.txt       # Classification threshold
├── datasets/                    # Example test datasets
├── embedding_examples/          # Example embeddings
│   ├── esm2/
│   ├── esm3/
│   └── esmc/
└── embedding_generation/        # Scripts to generate embeddings
    └── README.md                # Embedding generation guide
```

## Notes

- Protein names must match exactly across all embedding directories and FASTA files
- Only proteins with all three embeddings (ESM2, ESM3, ESMC) will be processed
- Use `--verbose` flag for detailed processing information
- GPU (CUDA) is automatically used if available

## Citation

If you use PRIMED in your research, please cite:
```
[Your citation here]
```

## Contact

For questions or issues:
- Please contact Luoshu Zhang at luoshu.zhang@ufl.edu
- GitHub: [Repository link]

## License

This project is licensed under the MIT License.
