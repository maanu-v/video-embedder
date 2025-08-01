"""
chunk_utils.py

- extract_frame_at_time(filepath, time): OpenCV snapshot
- extract_audio_segment(filepath, start, end): ffmpeg or moviepy
- save_chunk_assets(): saves image, audio, and metadata per chunk
"""

import os
import subprocess
import json
import uuid
import cv2
import numpy as np
from pathlib import Path
import sys

# Add the project root to sys.path to import from config
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CHUNKS_DIR

def extract_frame_at_time(filepath: str, time_seconds: float) -> np.ndarray:
    """
    Extract a frame from a video file at a specific time
    
    Args:
        filepath: Path to the video file
        time_seconds: Time in seconds at which to extract the frame
        
    Returns:
        numpy.ndarray: The extracted frame as a numpy array (BGR format)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Video file not found: {filepath}")
    
    # Open the video file
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {filepath}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Convert time to frame number
    frame_num = int(time_seconds * fps)
    
    # Ensure frame number is within bounds
    frame_num = max(0, min(frame_num, total_frames - 1))
    
    # Set position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if not ret:
        raise RuntimeError(f"Failed to extract frame at time {time_seconds} seconds")
    
    return frame

def extract_audio_segment(filepath: str, start_time: float, end_time: float, output_path: str = None) -> str:
    """
    Extract an audio segment from a video file using ffmpeg
    
    Args:
        filepath: Path to the video file
        start_time: Start time of the segment in seconds
        end_time: End time of the segment in seconds
        output_path: Path to save the extracted audio segment (if None, will generate a temporary file)
        
    Returns:
        Path to the extracted audio segment
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Video file not found: {filepath}")
    
    if output_path is None:
        output_path = CHUNKS_DIR / f"audio_segment_{uuid.uuid4()}.wav"
    else:
        output_path = Path(output_path)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate duration
    duration = end_time - start_time
    
    # Use ffmpeg to extract the audio segment
    cmd = [
        "ffmpeg",
        "-i", str(filepath),
        "-ss", str(start_time),
        "-t", str(duration),
        "-q:a", "0",  # High quality audio
        "-map", "a",  # Extract audio only
        "-y",  # Overwrite existing files without asking
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract audio segment: {result.stderr}")
    
    return str(output_path)

def save_chunk_assets(
    video_path: str, 
    chunk_id: str,
    start_time: float, 
    end_time: float,
    text: str,
    output_dir: Path = None,
    extract_frame: bool = True,
    extract_audio: bool = True
) -> dict:
    """
    Save assets for a video chunk (frame, audio, metadata)
    
    Args:
        video_path: Path to the video file
        chunk_id: Unique identifier for the chunk
        start_time: Start time of the chunk in seconds
        end_time: End time of the chunk in seconds
        text: Text transcript for the chunk
        output_dir: Directory to save chunk assets (if None, will use CHUNKS_DIR)
        extract_frame: Whether to extract a frame from the chunk
        extract_audio: Whether to extract the audio from the chunk
        
    Returns:
        Dictionary containing paths to the saved assets
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_dir is None:
        output_dir = CHUNKS_DIR / video_path.stem / chunk_id
    else:
        output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the result dictionary
    result = {
        "video_path": str(video_path),
        "chunk_id": chunk_id,
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "text": text
    }
    
    # Extract frame (take from middle of segment)
    if extract_frame:
        frame_time = (start_time + end_time) / 2
        frame = extract_frame_at_time(video_path, frame_time)
        
        # Save the frame as an image
        frame_path = output_dir / "frame.jpg"
        cv2.imwrite(str(frame_path), frame)
        result["frame_path"] = str(frame_path)
        result["frame_time"] = frame_time
    
    # Extract audio segment
    if extract_audio:
        audio_path = output_dir / "audio.wav"
        extracted_audio = extract_audio_segment(video_path, start_time, end_time, audio_path)
        result["audio_path"] = extracted_audio
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result
