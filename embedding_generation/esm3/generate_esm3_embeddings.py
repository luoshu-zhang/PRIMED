import os
import gc
import argparse
import torch
import torch.nn as nn
from Bio import SeqIO
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate ESM3 embeddings for protein sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input
    parser.add_argument('--input-files', type=str, nargs='+', required=True,
                        help='FASTA file(s) to process (space-separated for multiple)')
    
    # Model configuration
    parser.add_argument('--max-tokens', type=int, default=1022,
                        help='Maximum tokens per chunk')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./esm3_embeddings',
                        help='Directory to save embeddings')
    parser.add_argument('--output-prefix', type=str, default='',
                        help='Prefix for output filenames (optional)')
    
    # Processing options
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Apply z-score normalization to embeddings')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Disable normalization')
    parser.add_argument('--half-precision', action='store_true', default=True,
                        help='Use FP16 (half precision) for faster inference')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip proteins that already have embeddings')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed processing information')
    
    return parser.parse_args()


def read_fasta_sequences(fasta_path, verbose=False):
    """Parse FASTA file (standard 2-line format or 3-line with labels).
    
    Returns:
        list: [(header, sequence), ...]
    """
    records = []
    
    # Try standard BioPython parsing first
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            header = record.id
            sequence = str(record.seq)
            if len(sequence) > 0:
                records.append((header, sequence))
        return records
    except:
        pass
    
    # Fallback: manual parsing for 3-line format
    with open(fasta_path, "r") as f:
        lines = f.read().splitlines()
    
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        
        header = lines[i][1:].strip()
        
        if i + 1 >= len(lines):
            if verbose:
                print(f"   ⚠️ Skipping {header}: no sequence found")
            i += 1
            continue
        
        seq = lines[i + 1].strip()
        
        # Skip label line if present (3-line format)
        if i + 2 < len(lines) and not lines[i + 2].startswith(">"):
            potential_label = lines[i + 2].strip()
            if len(potential_label) == len(seq) and all(c in "01" for c in potential_label):
                i += 3
            else:
                i += 2
        else:
            i += 2
        
        if len(seq) > 0:
            records.append((header, seq))
    
    return records


def chunk_sequence(seq, chunk_size):
    """Split sequence into chunks of specified size.
    
    Args:
        seq: Protein sequence
        chunk_size: Maximum chunk length
    
    Returns:
        list: List of sequence chunks
    """
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]


def generate_embeddings(args):
    """Main function to generate ESM3 embeddings."""
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ESM3 model
    print(f"Loading ESM3 model (ESM3_OPEN_SMALL)...")
    client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
    
    if args.half_precision and device.type == 'cuda':
        client = client.half()
        print("Using FP16 (half precision)")
    
    print("✅ Model loaded")
    
    # Get embedding dimension
    print(f"Embedding dimension: 1536")
    print(f"Max tokens per chunk: {args.max_tokens}")
    print(f"Normalization: {'enabled' if args.normalize else 'disabled'}")
    print(f"Skip existing: {'enabled' if args.skip_existing else 'disabled'}\n")
    
    # Process all input files
    total_processed = 0
    total_skipped = 0
    total_already_exists = 0
    
    for fasta_path in args.input_files:
        print(f"{'='*80}")
        print(f"Processing: {fasta_path}")
        print(f"{'='*80}")
        
        if not os.path.exists(fasta_path):
            print(f"❌ File not found: {fasta_path}")
            continue
        
        # Load sequences
        records = read_fasta_sequences(fasta_path, args.verbose)
        print(f"Found {len(records)} sequences")
        
        if len(records) == 0:
            print(f"⚠️ No valid entries in {fasta_path}")
            continue
        
        # Process each protein
        for idx, (header, sequence) in enumerate(records, 1):
            try:
                # Check if already exists
                output_filename = f"{args.output_prefix}{header}.pt" if args.output_prefix else f"{header}.pt"
                output_path = os.path.join(args.output_dir, output_filename)
                
                if args.skip_existing and os.path.exists(output_path):
                    if args.verbose:
                        print(f"[{idx}/{len(records)}] ⏭️  Skipping {header} (already exists)")
                    total_already_exists += 1
                    continue
                
                if args.verbose:
                    print(f"\n[{idx}/{len(records)}] Processing: {header}")
                    print(f"   Sequence length: {len(sequence)}")
                elif idx % 10 == 0:
                    print(f"   Processed {idx}/{len(records)} sequences...")
                
                # Chunk if necessary
                chunks = chunk_sequence(sequence, args.max_tokens)
                
                if args.verbose and len(chunks) > 1:
                    print(f"   Chunking into {len(chunks)} parts")
                
                chunk_embeddings = []
                
                # Process each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    if args.verbose and len(chunks) > 1:
                        print(f"   Processing chunk {chunk_idx+1}/{len(chunks)} (length: {len(chunk)})")
                    
                    # Create protein and encode
                    protein = ESMProtein(sequence=chunk)
                    token_tensor = client.encode(protein)
                    
                    # Generate embeddings
                    output = client.forward_and_sample(
                        token_tensor,
                        SamplingConfig(return_per_residue_embeddings=True)
                    )
                    
                    # Extract embeddings (remove BOS/EOS tokens)
                    emb = output.per_residue_embedding[1:-1].cpu()
                    
                    # Normalize if requested
                    if args.normalize:
                        emb = (emb - emb.mean(dim=1, keepdim=True)) / (emb.std(dim=1, keepdim=True) + 1e-6)
                    
                    chunk_embeddings.append(emb)
                    
                    # Memory cleanup
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Concatenate all chunks
                full_embedding = torch.cat(chunk_embeddings, dim=0)
                
                # Save embedding
                torch.save({
                    'embedding': full_embedding,
                    'header': header,
                    'sequence_length': len(sequence)
                }, output_path)
                
                if args.verbose:
                    print(f"   ✅ Saved: {output_filename} | Shape: {full_embedding.shape}")
                
                total_processed += 1
                
                # Memory cleanup
                del full_embedding, chunk_embeddings
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
    print(f"Newly processed: {total_processed}")
    print(f"Already existed: {total_already_exists}")
    print(f"Skipped (errors): {total_skipped}")
    print(f"Output directory: {args.output_dir}")
    print(f"\n✅ Done!")


def main():
    args = parse_args()
    generate_embeddings(args)


if __name__ == "__main__":
    main()
