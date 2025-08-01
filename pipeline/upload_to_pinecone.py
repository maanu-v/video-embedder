"""
upload_to_pinecone.py

- Loads vectors + metadata
- Uses Pinecone's upsert() from new SDK (pinecone)
- Metadata includes: video_id, chunk_id, start_time, end_time, text, etc.
"""

import os
import json
import argparse
import time
from pathlib import Path
import sys
from typing import List, Dict, Any, Union
import pinecone

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    PINECONE_API_KEY, 
    PINECONE_ENVIRONMENT, 
    PINECONE_INDEX_NAME, 
    VECTOR_DIMENSION,
    BASE_DIR
)

# Directory with embeddings
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

def initialize_pinecone():
    """
    Initialize Pinecone connection and ensure index exists
    
    Returns:
        Pinecone index object
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if it doesn't
    indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in indexes:
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="cosine"
        )
        # Wait for index to be ready
        time.sleep(5)
    
    # Connect to the index
    index = pc.Index(PINECONE_INDEX_NAME)
    return index

def prepare_records_for_pinecone(embeddings_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepare embedding records for Pinecone upload
    
    Args:
        embeddings_data: Dictionary containing embeddings data
        
    Returns:
        List of records formatted for Pinecone upsert
    """
    video_id = embeddings_data.get('video_id')
    chunks = embeddings_data.get('chunks', [])
    
    records = []
    for chunk in chunks:
        # Extract embedding
        embedding = chunk.get('embedding')
        if not embedding:
            continue
        
        # Create a unique ID for the vector
        chunk_id = chunk.get('chunk_id')
        vector_id = f"{video_id}_{chunk_id}"
        
        # Prepare metadata (exclude the embedding itself to save space)
        metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
        
        # Ensure all metadata values are strings, numbers, or booleans
        # Pinecone has limitations on metadata types
        # Ensure all metadata values are strings, numbers, or booleans
        for key, value in metadata.items():
            if isinstance(value, Path):
                metadata[key] = str(value)
            elif isinstance(value, (list, dict)):
                metadata[key] = json.dumps(value)

        # Create record
        record = {
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        }
        
        records.append(record)
    
    return records

def upload_to_pinecone(embeddings_file: str, batch_size: int = 100) -> int:
    """
    Upload embeddings to Pinecone
    
    Args:
        embeddings_file: Path to the embeddings JSON file
        batch_size: Number of vectors to upload in each batch
        
    Returns:
        Number of vectors uploaded
    """
    embeddings_file = Path(embeddings_file)
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    # Load embeddings
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)
    
    # Initialize Pinecone
    index = initialize_pinecone()
    
    # Prepare records for upload
    records = prepare_records_for_pinecone(embeddings_data)
    
    print(f"Uploading {len(records)} vectors to Pinecone index: {PINECONE_INDEX_NAME}")
    
    # Upload in batches
    total_uploaded = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            total_uploaded += len(batch)
            print(f"Uploaded batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}: {total_uploaded}/{len(records)} vectors")
        except Exception as e:
            print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
    
    print(f"Upload completed. Total vectors uploaded: {total_uploaded}")
    
    return total_uploaded

def main():
    parser = argparse.ArgumentParser(description="Upload embeddings to Pinecone")
    parser.add_argument("embeddings", nargs="?", help="Path to embeddings JSON file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for uploading")
    args = parser.parse_args()
    
    # Process a single embeddings file if specified
    if args.embeddings:
        embeddings_file = args.embeddings
        upload_to_pinecone(embeddings_file, args.batch_size)
    else:
        # Process all embeddings files in the embeddings directory
        for embeddings_file in EMBEDDINGS_DIR.glob("*_embeddings.json"):
            try:
                upload_to_pinecone(str(embeddings_file), args.batch_size)
            except Exception as e:
                print(f"Error processing {embeddings_file}: {str(e)}")

if __name__ == "__main__":
    main()
