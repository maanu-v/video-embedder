"""
run_transcription.py

- Takes video from /data/raw_videos/
- Calls transcribe_video() and maps sentence timestamps
- Outputs: /data/processed/video_id_transcript.json
"""

import os
import json
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.transcript_utils import transcribe_video, process_transcript, handle_non_transcribed_segments
from utils.video_utils import load_video_metadata
from config.settings import RAW_VIDEOS_DIR, PROCESSED_DIR

def run_transcription(video_path: str, output_dir: Path = None) -> Path:
    """
    Run transcription on a video file and process the transcript
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the processed transcript (if None, uses PROCESSED_DIR)
    
    Returns:
        Path to the processed transcript file
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file path
    transcript_file = output_dir / f"{video_path.stem}_transcript.json"
    processed_file = output_dir / f"{video_path.stem}_processed.json"
    
    print(f"Transcribing video: {video_path}")
    
    # Get video metadata for duration info
    video_metadata = load_video_metadata(str(video_path))
    video_duration = video_metadata.get('duration', 0)
    
    # Run transcription with Whisper
    transcript_result = transcribe_video(str(video_path), str(transcript_file))
    
    print(f"Transcription completed. Processing sentences...")
    
    # Process transcript to get sentence-level data
    sentence_data = process_transcript(transcript_result)
    
    # Handle segments with no transcription
    processed_data = handle_non_transcribed_segments(sentence_data, video_duration)
    
    # Save processed data
    with open(processed_file, 'w') as f:
        json.dump({
            'video_id': video_path.stem,
            'video_path': str(video_path),
            'video_duration': video_duration,
            'sentences': processed_data
        }, f, indent=2)
    
    print(f"Processing completed. Found {len(processed_data)} segments.")
    print(f"Saved processed transcript to {processed_file}")
    
    return processed_file

def main():
    parser = argparse.ArgumentParser(description="Transcribe video and process into sentence segments")
    parser.add_argument("video", nargs="?", help="Path to video file (if not provided, will process all videos in RAW_VIDEOS_DIR)")
    parser.add_argument("--output-dir", help="Directory to save processed transcripts")
    args = parser.parse_args()
    
    # Process a single video if specified
    if args.video:
        video_path = args.video
        run_transcription(video_path, args.output_dir)
    else:
        # Process all videos in the raw_videos directory
        for video_file in RAW_VIDEOS_DIR.glob("*"):
            if video_file.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
                try:
                    run_transcription(str(video_file), args.output_dir)
                except Exception as e:
                    print(f"Error processing {video_file}: {str(e)}")

if __name__ == "__main__":
    main()
