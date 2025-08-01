"""
generate_chunks.py

- For each sentence in transcript:
    - Extract frame at sentence start
    - Extract audio between start and end
    - Save outputs to /data/processed/video_id/chunk_{id}/
"""

import os
import json
import argparse
import uuid
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.chunk_utils import save_chunk_assets
from config.settings import PROCESSED_DIR, CHUNKS_DIR

def generate_chunks(processed_transcript_path: str, output_dir: Path = None) -> list:
    """
    Generate video and audio chunks for each sentence in a processed transcript
    
    Args:
        processed_transcript_path: Path to the processed transcript JSON file
        output_dir: Directory to save the chunk assets (if None, uses CHUNKS_DIR)
        
    Returns:
        List of dictionaries containing metadata for each chunk
    """
    processed_transcript_path = Path(processed_transcript_path)
    if not processed_transcript_path.exists():
        raise FileNotFoundError(f"Processed transcript file not found: {processed_transcript_path}")
    
    # Load the processed transcript
    with open(processed_transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    video_id = transcript_data.get('video_id', processed_transcript_path.stem)
    video_path = transcript_data.get('video_path')
    sentences = transcript_data.get('sentences', [])
    
    if not video_path or not Path(video_path).exists():
        raise ValueError(f"Video file not found: {video_path}")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = CHUNKS_DIR / video_id
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating chunks for video: {video_id}")
    print(f"Found {len(sentences)} sentences to process")
    
    # Generate chunks for each sentence
    chunk_data = []
    for i, sentence in enumerate(sentences):
        # Generate a unique chunk ID
        chunk_id = f"chunk_{i:04d}"
        
        # Get sentence data
        start_time = sentence.get('start', 0)
        end_time = sentence.get('end', 0)
        text = sentence.get('sentence', '')
        is_silent = sentence.get('is_silent', False)
        
        print(f"Processing {chunk_id}: {start_time:.2f}s - {end_time:.2f}s")
        
        # Save chunk assets (frame and audio)
        chunk_assets = save_chunk_assets(
            video_path=video_path,
            chunk_id=chunk_id,
            start_time=start_time,
            end_time=end_time,
            text=text,
            output_dir=output_dir / chunk_id
        )
        
        # Add chunk metadata
        chunk_assets['video_id'] = video_id
        chunk_assets['chunk_index'] = i
        chunk_assets['is_silent'] = is_silent
        
        chunk_data.append(chunk_assets)
    
    # Save chunk metadata summary
    chunks_summary_path = output_dir / "chunks_summary.json"
    with open(chunks_summary_path, 'w') as f:
        json.dump({
            'video_id': video_id,
            'video_path': video_path,
            'total_chunks': len(chunk_data),
            'chunks': chunk_data
        }, f, indent=2)
    
    print(f"Chunk generation completed. Generated {len(chunk_data)} chunks.")
    print(f"Saved chunks summary to {chunks_summary_path}")
    
    return chunk_data

def main():
    parser = argparse.ArgumentParser(description="Generate video chunks from processed transcript")
    parser.add_argument("transcript", nargs="?", help="Path to processed transcript JSON file")
    parser.add_argument("--output-dir", help="Directory to save chunk assets")
    args = parser.parse_args()
    
    # Process a single transcript file if specified
    if args.transcript:
        transcript_path = args.transcript
        generate_chunks(transcript_path, args.output_dir)
    else:
        # Process all transcript files in the processed directory
        for transcript_file in PROCESSED_DIR.glob("*_processed.json"):
            try:
                generate_chunks(str(transcript_file), args.output_dir)
            except Exception as e:
                print(f"Error processing {transcript_file}: {str(e)}")

if __name__ == "__main__":
    main()
