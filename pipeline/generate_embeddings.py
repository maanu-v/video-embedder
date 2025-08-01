"""
generate_embeddings.py

- For each chunk:
    - Load frame + sentence text
    - Call embed_frame_with_text()
    - Save vector locally in /data/embeddings/ or return directly
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.embedding_utils import embed_frame_with_text, embed_text
from config.settings import CHUNKS_DIR, PROCESSED_DIR, BASE_DIR

# Directory to store embeddings
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

def generate_embedding_for_chunk(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a multimodal embedding for a single chunk
    
    Args:
        chunk_data: Dictionary containing chunk metadata
        
    Returns:
        Dictionary containing chunk metadata and embedding
    """
    # Get required data from chunk
    frame_path = chunk_data.get('frame_path')
    text = chunk_data.get('text', '')
    
    if not frame_path or not Path(frame_path).exists():
        print(f"Warning: Frame file not found: {frame_path}. Using text-only embedding.")
        # Fallback to text-only embedding
        embedding = embed_text(text)
    else:
        # Generate multimodal embedding
        embedding = embed_frame_with_text(frame_path, text)
    
    # Add embedding to chunk data
    chunk_data['embedding'] = embedding.tolist()
    
    return chunk_data

def generate_embeddings(chunks_summary_path: str, output_dir: Path = None) -> Path:
    """
    Generate embeddings for all chunks in a summary file
    
    Args:
        chunks_summary_path: Path to the chunks summary JSON file
        output_dir: Directory to save embeddings (if None, uses EMBEDDINGS_DIR)
        
    Returns:
        Path to the embeddings file
    """
    chunks_summary_path = Path(chunks_summary_path)
    if not chunks_summary_path.exists():
        raise FileNotFoundError(f"Chunks summary file not found: {chunks_summary_path}")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = EMBEDDINGS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load chunks summary
    with open(chunks_summary_path, 'r') as f:
        summary_data = json.load(f)
    
    video_id = summary_data.get('video_id')
    chunks = summary_data.get('chunks', [])
    
    print(f"Generating embeddings for video: {video_id}")
    print(f"Found {len(chunks)} chunks to process")
    
    # Generate embeddings for each chunk
    start_time = time.time()
    embedded_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}: {chunk.get('chunk_id')}")
        try:
            embedded_chunk = generate_embedding_for_chunk(chunk)
            embedded_chunks.append(embedded_chunk)
        except Exception as e:
            print(f"Error generating embedding for chunk {chunk.get('chunk_id')}: {str(e)}")
    
    # Save embeddings
    embeddings_file = output_dir / f"{video_id}_embeddings.json"
    with open(embeddings_file, 'w') as f:
        json.dump({
            'video_id': video_id,
            'total_chunks': len(embedded_chunks),
            'embedding_dimension': len(embedded_chunks[0]['embedding']) if embedded_chunks else 0,
            'chunks': embedded_chunks
        }, f, indent=2)
    
    elapsed_time = time.time() - start_time
    print(f"Embedding generation completed in {elapsed_time:.2f} seconds.")
    print(f"Generated {len(embedded_chunks)} embeddings.")
    print(f"Saved embeddings to {embeddings_file}")
    
    return embeddings_file

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for video chunks")
    parser.add_argument("summary", nargs="?", help="Path to chunks summary JSON file")
    parser.add_argument("--output-dir", help="Directory to save embeddings")
    args = parser.parse_args()
    
    # Process a single summary file if specified
    if args.summary:
        summary_path = args.summary
        generate_embeddings(summary_path, args.output_dir)
    else:
        # Process all summary files in the chunks directory
        for summary_file in CHUNKS_DIR.glob("*/chunks_summary.json"):
            try:
                generate_embeddings(str(summary_file), args.output_dir)
            except Exception as e:
                print(f"Error processing {summary_file}: {str(e)}")

if __name__ == "__main__":
    main()
