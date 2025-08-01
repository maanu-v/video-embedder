"""
main.py

Runs the full pipeline:
1. Transcribe video using Whisper
2. Split transcript into sentence-timed chunks
3. Extract frame and audio per sentence chunk
4. Embed each chunk using Gemini Pro
5. Upload embeddings to Pinecone with metadata
"""

import os
import argparse
import shutil
from pathlib import Path
import sys
import time

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from pipeline.run_transcription import run_transcription
from pipeline.generate_chunks import generate_chunks
from pipeline.generate_embeddings import generate_embeddings
from pipeline.upload_to_pinecone import upload_to_pinecone
from config.settings import RAW_VIDEOS_DIR, PROCESSED_DIR, CHUNKS_DIR, BASE_DIR

def cleanup_old_data(video_name: str, cleanup_chunks: bool = True, cleanup_embeddings: bool = True) -> None:
    """
    Clean up old data for a video to ensure fresh processing
    
    Args:
        video_name: Name of the video (stem)
        cleanup_chunks: Whether to remove old chunks
        cleanup_embeddings: Whether to remove old embeddings
    """
    if cleanup_chunks:
        # Remove old chunks directory
        chunk_dir = CHUNKS_DIR / video_name
        if chunk_dir.exists():
            print(f"Removing old chunks directory: {chunk_dir}")
            shutil.rmtree(chunk_dir)
    
    if cleanup_embeddings:
        # Remove old embeddings file
        embedding_file = BASE_DIR / "data" / "embeddings" / f"{video_name}_embeddings.json"
        if embedding_file.exists():
            print(f"Removing old embeddings file: {embedding_file}")
            embedding_file.unlink()

def process_video(video_path: str, skip_upload: bool = False, force_reprocess: bool = False) -> None:
    """
    Process a single video through the full pipeline
    
    Args:
        video_path: Path to the video file
        skip_upload: If True, skip uploading to Pinecone
        force_reprocess: If True, force reprocessing of all steps
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"\n{'='*80}")
    print(f"PROCESSING VIDEO: {video_path}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Clean up old data if force_reprocess
    if force_reprocess:
        cleanup_old_data(video_path.stem)
    
    # Step 1: Transcribe video
    print(f"\n[STEP 1/4] Transcribing video...")
    transcript_path = run_transcription(str(video_path))
    
    # Step 2: Generate chunks
    print(f"\n[STEP 2/4] Generating video chunks...")
    chunks_summary_path = CHUNKS_DIR / video_path.stem / "chunks_summary.json"
    if not chunks_summary_path.exists() or force_reprocess:
        generate_chunks(str(transcript_path))
    else:
        print(f"Using existing chunks summary at {chunks_summary_path}")
    
    # Step 3: Generate embeddings
    print(f"\n[STEP 3/4] Generating embeddings...")
    embeddings_file = BASE_DIR / "data" / "embeddings" / f"{video_path.stem}_embeddings.json"
    if not embeddings_file.exists() or force_reprocess:
        generate_embeddings(str(chunks_summary_path))
    else:
        print(f"Using existing embeddings at {embeddings_file}")
    
    # Step 4: Upload to Pinecone
    if not skip_upload:
        print(f"\n[STEP 4/4] Uploading embeddings to Pinecone...")
        upload_to_pinecone(str(embeddings_file))
    else:
        print(f"\n[STEP 4/4] Skipping upload to Pinecone (--skip-upload flag used)")
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETED in {elapsed_time:.2f} seconds")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Run the full video processing pipeline")
    parser.add_argument("video", nargs="?", help="Path to video file (if not provided, will process all videos in data/raw_videos/)")
    parser.add_argument("--skip-transcription", action="store_true", help="Skip transcription step if transcript already exists")
    parser.add_argument("--skip-chunks", action="store_true", help="Skip chunk generation step if chunks already exist")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation step if embeddings already exist")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading to Pinecone")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all steps")
    parser.add_argument("--process-all", action="store_true", help="Process all videos including those already processed")
    args = parser.parse_args()
    
    # Import the video processing utilities
    from utils.check_processed_utils import is_video_processed, mark_video_processed, get_unprocessed_videos
    
    # Process a single video if specified
    if args.video:
        video_path = args.video
        # Check if the video has already been processed
        if not args.force and not args.process_all and is_video_processed(video_path):
            print(f"Video {video_path} has already been processed. Use --force or --process-all to reprocess.")
        else:
            process_video(video_path, args.skip_upload, args.force)
            # Mark the video as processed
            mark_video_processed(video_path)
    else:
        # Get videos to process based on processed status
        if args.process_all or args.force:
            # Process all videos regardless of status
            videos_to_process = [str(video_file) for video_file in RAW_VIDEOS_DIR.glob("*") 
                                if video_file.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']]
        else:
            # Only process unprocessed videos
            videos_to_process = get_unprocessed_videos(RAW_VIDEOS_DIR)
            
        if not videos_to_process:
            print("No new videos to process. Use --process-all or --force to reprocess existing videos.")
            return
            
        # Process selected videos
        video_count = 0
        for video_path in videos_to_process:
            try:
                process_video(video_path, args.skip_upload, args.force)
                mark_video_processed(video_path)
                video_count += 1
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
        
        if video_count == 0:
            print(f"No videos found in {RAW_VIDEOS_DIR}")
            print("Please place your videos in this directory or specify a video path.")

if __name__ == "__main__":
    main()
