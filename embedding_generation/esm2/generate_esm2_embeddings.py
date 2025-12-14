import os
import gc
import argparse
import torch
from esm import pretrained
from esm.data import BatchConverter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate ESM2 embeddings for protein sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input
    parser.add_argument('--input-files', type=str, nargs='+', required=True,
                        help='FASTA file(s) to process (space-separated for multiple)')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, 
                        default='esm2_t36_3B_UR50D',
                        choices=['esm2_t48_15B_UR50D', 'esm2_t36_3B_UR50D', 
                                'esm2_t33_650M_UR50D', 'esm2_t30_150M_UR50D',
                                'esm2_t12_35M_UR50D', 'esm2_t6_8M_UR50D'],
                        help='ESM2 model variant')
    parser.add_argument('--max-tokens', type=int, default=1022,
                        help='Maximum tokens per chunk (ESM2 limit: 1022)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./esm2_embeddings',
                        help='Directory to save embeddings')
    parser.add_argument('--output-prefix', type=str, default='',
                        help='Prefix for output filenames (optional)')
    
    # Processing options
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Apply z-score normalization to embeddings')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Disable normalization')
    parser.add_argument('--save-labels', action='store_true', default=True,
                        help='Save labels in output (if available in FASTA)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed processing information')
    
    return parser.parse_args()


def load_fasta_entries(fasta_path, verbose=False):
    """Parse FASTA file with optional 3-line format (header, sequence, labels).
    
    Supports two formats:
    1. Standard FASTA (2 lines): >header, sequence
    2. Extended FASTA (3 lines): >header, sequence, binary labels
    
    Returns:
        list: [(header, sequence, label_or_None), ...]
    """
    entries = []
    with open(fasta_path, "r") as f:
        lines = f.read().splitlines()
    
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        
        header = lines[i][1:].strip()
        
        # Check if we have sequence
        if i + 1 >= len(lines):
            if verbose:
                print(f"   ⚠️ Skipping {header}: no sequence found")
            i += 1
            continue
        
        seq = lines[i + 1].strip()
        
        # Check if we have labels (3-line format)
        label = None
        if i + 2 < len(lines) and not lines[i + 2].startswith(">"):
            potential_label = lines[i + 2].strip()
            # Validate it's a binary label line
            if len(potential_label) == len(seq) and all(c in "01" for c in potential_label):
                label = potential_label
                i += 3
            else:
                i += 2
        else:
            i += 2
        
        # Validate sequence
        if len(seq) == 0:
            if verbose:
                print(f"   ⚠️ Skipping {header}: empty sequence")
            continue
        
        entries.append((header, seq, label))
    
    return entries


def chunk_sequence(seq, label, max_len):
    """Split sequence and labels into chunks that fit within max_len.
    
    Args:
        seq: Protein sequence
        label: Binary label string or None
        max_len: Maximum chunk length
    
    Returns:
        list: [(seq_chunk, label_chunk_or_None), ...]
    """
    # Find optimal number of chunks
    for n in range(1, len(seq) + 1):
        chunk_len = (len(seq) + n - 1) // n
        if chunk_len <= max_len:
            chunks = []
            for start in range(0, len(seq), chunk_len):
                seq_chunk = seq[start:start + chunk_len]
                label_chunk = label[start:start + chunk_len] if label else None
                chunks.append((seq_chunk, label_chunk))
            return chunks
    
    # Fallback: single chunk (shouldn't reach here)
    return [(seq, label)]


def generate_embeddings(args):
    """Main function to generate ESM2 embeddings."""
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ESM2 model
    print(f"Loading ESM2 model: {args.model_name}...")
    model, alphabet = pretrained.load_model_and_alphabet(args.model_name)
    model.eval().to(device)
    batch_converter = BatchConverter(alphabet)
    print("✅ Model loaded")
    
    # Get embedding dimension
    with torch.no_grad():
        test_seq = "M"
        _, _, test_toks = batch_converter([("test", test_seq)])
        test_toks = test_toks.to(device)
        test_out = model(test_toks, repr_layers=[model.num_layers])
        embedding_dim = test_out["representations"][model.num_layers].shape[-1]
    
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Max tokens per chunk: {args.max_tokens}")
    print(f"Normalization: {'enabled' if args.normalize else 'disabled'}\n")
    
    # Process all input files
    total_processed = 0
    total_skipped = 0
    
    for fasta_path in args.input_files:
        print(f"{'='*80}")
        print(f"Processing: {fasta_path}")
        print(f"{'='*80}")
        
        if not os.path.exists(fasta_path):
            print(f"❌ File not found: {fasta_path}")
            continue
        
        # Load entries
        entries = load_fasta_entries(fasta_path, args.verbose)
        print(f"Found {len(entries)} sequences")
        
        if len(entries) == 0:
            print(f"⚠️ No valid entries in {fasta_path}")
            continue
        
        # Check if labels are present
        has_labels = any(label is not None for _, _, label in entries)
        if has_labels:
            print("Labels detected in FASTA file")
        
        # Process each protein
        for idx, (header, seq, label_line) in enumerate(entries, 1):
            try:
                if args.verbose:
                    print(f"\n[{idx}/{len(entries)}] Processing: {header}")
                    print(f"   Sequence length: {len(seq)}")
                
                # Chunk if necessary
                chunks = chunk_sequence(seq, label_line, args.max_tokens)
                
                if args.verbose and len(chunks) > 1:
                    print(f"   Chunking into {len(chunks)} parts")
                
                all_embs = []
                all_labels = []
                
                # Process each chunk
                for chunk_idx, (seq_chunk, label_chunk) in enumerate(chunks):
                    # Tokenize
                    _, _, toks = batch_converter([(f"{header}_chunk{chunk_idx}", seq_chunk)])
                    toks = toks.to(device)
                    
                    # Generate embedding
                    with torch.no_grad():
                        out = model(toks, repr_layers=[model.num_layers])
                        emb = out["representations"][model.num_layers][0, 1:1 + len(seq_chunk)]
                    
                    # Normalize if requested
                    if args.normalize:
                        emb = (emb - emb.mean(dim=1, keepdim=True)) / (emb.std(dim=1, keepdim=True) + 1e-6)
                    
                    all_embs.append(emb.cpu())
                    
                    # Process labels if available
                    if label_chunk is not None and args.save_labels:
                        label_tensor = torch.tensor(
                            [int(c) for c in label_chunk], 
                            dtype=torch.float32
                        ).unsqueeze(1)
                        all_labels.append(label_tensor)
                
                # Concatenate all chunks
                emb_cat = torch.cat(all_embs, dim=0)
                
                # Prepare output dictionary
                output_dict = {
                    "embedding": emb_cat,
                    "header": header
                }
                
                # Add labels if available and requested
                if len(all_labels) > 0:
                    label_cat = torch.cat(all_labels, dim=0)
                    output_dict["label"] = label_cat
                
                # Save
                output_filename = f"{args.output_prefix}{header}.pt" if args.output_prefix else f"{header}.pt"
                output_path = os.path.join(args.output_dir, output_filename)
                torch.save(output_dict, output_path)
                
                if args.verbose:
                    print(f"   ✅ Saved: {output_filename}")
                elif idx % 10 == 0:
                    print(f"   Processed {idx}/{len(entries)} sequences...")
                
                total_processed += 1
                
                # Memory cleanup
                del emb_cat, all_embs, output_dict
                if len(all_labels) > 0:
                    del label_cat, all_labels
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ Error processing {header}: {e}")
                total_skipped += 1
                continue
        
        print(f"\n✅ Completed {fasta_path}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total sequences processed: {total_processed}")
    print(f"Total sequences skipped: {total_skipped}")
    print(f"Output directory: {args.output_dir}")
    print(f"\n✅ Done!")


def main():
    args = parse_args()
    generate_embeddings(args)


if __name__ == "__main__":
    main()
