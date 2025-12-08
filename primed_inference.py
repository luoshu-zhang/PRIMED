import os
import gc
import json
import argparse
import torch
import numpy as np
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
from torch.nn.functional import sigmoid
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron for DNA-binding prediction."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PRIMED: DNA-binding prediction using protein language model embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input paths - Embeddings
    parser.add_argument('--esm2-dir', type=str, required=True,
                        help='Directory containing ESM2 embeddings (.pt files)')
    parser.add_argument('--esm3-dir', type=str, required=True,
                        help='Directory containing ESM3 embeddings (.pt files)')
    parser.add_argument('--esmc-dir', type=str, required=True,
                        help='Directory containing ESMC embeddings (.pt files)')
    
    # Input paths - Test datasets
    parser.add_argument('--test-datasets', type=str, nargs='+', required=True,
                        help='FASTA files for test datasets (can specify multiple)')
    parser.add_argument('--dataset-names', type=str, nargs='+', default=None,
                        help='Names for datasets (default: auto-generated from filenames)')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained PRIMED model (.pth file)')
    parser.add_argument('--input-dim', type=int, default=5248,
                        help='Input dimension (concatenated embedding size)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (if not provided, reads from threshold file)')
    parser.add_argument('--threshold-file', type=str, default='optimized_threshold.txt',
                        help='File containing threshold (used if --threshold not specified)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save output files')
    parser.add_argument('--output-prefix', type=str, default='primed',
                        help='Prefix for output filenames')
    parser.add_argument('--save-scores', action='store_true',
                        help='Save raw prediction scores (probabilities) to JSON')
    
    # Processing options
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed processing information')
    
    return parser.parse_args()


def parse_labels_from_fasta(fasta_file):
    """Parse binary labels from FASTA file.
    
    Expected format:
    >header
    SEQUENCE
    010101... (binary labels)
    """
    labels_dict = {}
    with open(fasta_file, "r") as f:
        lines = f.read().splitlines()
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
            header = lines[i][1:].strip()
            label_line = lines[i + 2].strip()
            labels_dict[header] = [int(c) for c in label_line]
    return labels_dict


def get_fasta_headers(fasta_file):
    """Extract protein headers from FASTA file."""
    return [record.id for record in SeqIO.parse(fasta_file, "fasta")]


def run_primed_inference(esm2_dir, esm3_dir, esmc_dir, labels_dict, headers, 
                         model, threshold, device, dataset_name, verbose=False):
    """Run PRIMED inference on a dataset.
    
    Returns:
        tuple: (metrics_dict, predictions_dict)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}")
        print(f"   FASTA headers: {len(headers)}")
    
    # Get embedding headers
    esm2_headers = set(f.split(".pt")[0] for f in os.listdir(esm2_dir) if f.endswith(".pt"))
    esm3_headers = set(f.split(".pt")[0] for f in os.listdir(esm3_dir) if f.endswith(".pt"))
    esmc_headers = set(f.split(".pt")[0] for f in os.listdir(esmc_dir) if f.endswith(".pt"))
    
    if verbose:
        print(f"   ESM2 embeddings: {len(esm2_headers)}")
        print(f"   ESM3 embeddings: {len(esm3_headers)}")
        print(f"   ESMC embeddings: {len(esmc_headers)}")
    
    # Find valid proteins with all embeddings
    valid_headers = sorted(set(headers) & esm2_headers & esm3_headers & esmc_headers)
    
    if verbose:
        print(f"   Valid proteins: {len(valid_headers)}")
    
    if len(valid_headers) == 0:
        print(f"   ❌ ERROR: No proteins with all embeddings found!")
        return {}, {}
    
    all_preds, all_labels = [], []
    predictions_dict = {}
    processed = 0
    skipped = 0
    
    for header in valid_headers:
        try:
            # Load embeddings
            emb2 = torch.load(os.path.join(esm2_dir, f"{header}.pt"))["embedding"]
            emb3 = torch.load(os.path.join(esm3_dir, f"{header}.pt"))["embedding"]
            embc = torch.load(os.path.join(esmc_dir, f"{header}.pt"))["emb"]
            label = labels_dict[header]
            
            # Validate shapes
            if not (emb2.shape[0] == emb3.shape[0] == embc.shape[0] == len(label)):
                if verbose:
                    print(f"   ⚠️ Skipping {header}: shape mismatch")
                skipped += 1
                continue
            
            # Concatenate and predict
            input_tensor = torch.cat([emb2, emb3, embc], dim=1).to(device)
            with torch.no_grad():
                logits = model(input_tensor).squeeze()
                probs = sigmoid(logits).cpu().numpy()
            
            # Handle single residue case
            if isinstance(probs, np.ndarray) and probs.ndim == 0:
                probs = np.array([probs.item()])
            
            all_preds.extend(probs.tolist())
            all_labels.extend(label)
            predictions_dict[header] = probs.tolist()
            
            processed += 1
            
            # Memory management
            del emb2, emb3, embc, input_tensor, logits, probs
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Error on {header}: {e}")
            skipped += 1
            continue
    
    if verbose:
        print(f"   Processed: {processed}, Skipped: {skipped}")
        print(f"   Total residues: {len(all_preds):,}")
    
    if len(all_preds) == 0:
        print(f"   ❌ ERROR: No predictions generated!")
        return {}, {}
    
    # Calculate metrics
    predictions = np.array(all_preds)
    labels = np.array(all_labels)
    
    auc = roc_auc_score(labels, predictions)
    preds_bin = (predictions >= threshold).astype(int)
    mcc = matthews_corrcoef(labels, preds_bin)
    tn, fp, fn, tp = confusion_matrix(labels, preds_bin).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate binding percentage
    binding_pct = (np.sum(labels) / len(labels)) * 100 if len(labels) > 0 else 0
    
    metrics = {
        'dataset': dataset_name,
        'auc': auc,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'threshold': threshold,
        'n_proteins': processed,
        'n_residues': len(all_preds),
        'n_binding': int(np.sum(labels)),
        'binding_pct': binding_pct
    }
    
    if verbose:
        print(f"   AUC: {auc:.4f}, MCC: {mcc:.4f}, "
              f"Sen: {sensitivity:.4f}, Spe: {specificity:.4f}")
    
    return metrics, predictions_dict


def save_metrics_table(all_metrics, output_path, verbose=False):
    """Save metrics to a tab-delimited file."""
    with open(output_path, 'w') as f:
        f.write("Dataset\tAUC\tMCC\tSensitivity\tSpecificity\tThreshold\t"
                "Proteins\tResidues\tBinding_Sites\tBinding_%\n")
        for m in all_metrics:
            f.write(f"{m['dataset']}\t{m['auc']:.4f}\t{m['mcc']:.4f}\t"
                   f"{m['sensitivity']:.4f}\t{m['specificity']:.4f}\t{m['threshold']:.4f}\t"
                   f"{m['n_proteins']}\t{m['n_residues']}\t{m['n_binding']}\t{m['binding_pct']:.2f}\n")
    
    if verbose:
        print(f"   Saved: {output_path}")


def save_scores_json(scores_dict, output_path, verbose=False):
    """Save raw prediction scores to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(scores_dict, f, indent=2)
    
    if verbose:
        print(f"   Saved: {output_path}")


def main():
    args = parse_args()
    
    # Validate inputs
    if args.dataset_names is not None:
        if len(args.dataset_names) != len(args.test_datasets):
            print("❌ Error: Number of dataset names must match number of test datasets")
            return
    else:
        # Auto-generate names from filenames
        args.dataset_names = [
            os.path.splitext(os.path.basename(f))[0] 
            for f in args.test_datasets
        ]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    # Load threshold
    if args.threshold is None:
        try:
            with open(args.threshold_file, "r") as f:
                threshold = float(f.read().strip())
            print(f"Threshold: {threshold:.4f} (from {args.threshold_file})")
        except FileNotFoundError:
            threshold = 0.5
            print(f"Warning: {args.threshold_file} not found, using default: {threshold}")
    else:
        threshold = args.threshold
        print(f"Threshold: {threshold:.4f}")
    
    # Load model
    print(f"Loading model: {args.model_path}")
    model = MLP(args.input_dim, 1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Process all datasets
    print(f"\n{'='*80}")
    print(f"Processing {len(args.test_datasets)} dataset(s)")
    print(f"{'='*80}")
    
    all_metrics = []
    all_scores = {}
    
    for fasta_file, dataset_name in zip(args.test_datasets, args.dataset_names):
        # Parse labels and headers
        labels_dict = parse_labels_from_fasta(fasta_file)
        headers = get_fasta_headers(fasta_file)
        
        # Run inference
        metrics, scores = run_primed_inference(
            args.esm2_dir, args.esm3_dir, args.esmc_dir, labels_dict, headers,
            model, threshold, device, dataset_name, args.verbose
        )
        
        if metrics:
            all_metrics.append(metrics)
            if args.save_scores:
                all_scores[dataset_name] = scores
    
    # Print summary table
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'AUC':>7} {'MCC':>7} {'Sens':>7} {'Spec':>7} {'Proteins':>9}")
    print("-"*80)
    for m in all_metrics:
        print(f"{m['dataset']:<20} {m['auc']:>7.3f} {m['mcc']:>7.3f} "
              f"{m['sensitivity']:>7.3f} {m['specificity']:>7.3f} {m['n_proteins']:>9}")
    print("="*80)
    
    # Save outputs
    print(f"\nSaving outputs to: {args.output_dir}/")
    
    # Always save metrics
    metrics_path = os.path.join(args.output_dir, f"{args.output_prefix}_metrics.txt")
    save_metrics_table(all_metrics, metrics_path, args.verbose)
    print(f"  ✅ Metrics: {metrics_path}")
    
    # Optionally save scores
    if args.save_scores:
        for dataset_name, scores in all_scores.items():
            safe_name = dataset_name.replace(' ', '_').replace('/', '_')
            score_path = os.path.join(
                args.output_dir, 
                f"{args.output_prefix}_scores_{safe_name}.json"
            )
            save_scores_json(scores, score_path, args.verbose)
            print(f"  ✅ Scores: {score_path}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
