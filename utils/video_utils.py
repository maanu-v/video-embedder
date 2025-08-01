"""
video_utils.py

- load_video_metadata(filepath): get duration, resolution, fps
- extract_audio(filepath): extracts audio using moviepy or ffmpeg
"""

import os
import json
import subprocess
from pathlib import Path
import cv2
from typing import Dict, Tuple, Any

def load_video_metadata(filepath: str) -> Dict[str, Any]:
    """
    Get video metadata using ffprobe
    
    Args:
        filepath: Path to the video file
        
    Returns:
        Dict containing video metadata (duration, resolution, fps, etc)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Video file not found: {filepath}")
    
    # Use ffprobe to get video metadata in JSON format
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration:stream=width,height,r_frame_rate", 
        "-of", "json", 
        str(filepath)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get video metadata: {result.stderr}")
    
    metadata = json.loads(result.stdout)
    
    # Extract relevant information
    video_info = {}
    
    # Get duration from format section
    if 'format' in metadata and 'duration' in metadata['format']:
        video_info['duration'] = float(metadata['format']['duration'])
    
    # Find video stream and extract width, height, and fps
    if 'streams' in metadata:
        for stream in metadata['streams']:
            if 'width' in stream and 'height' in stream:  # This is a video stream
                video_info['width'] = stream['width']
                video_info['height'] = stream['height']
                
                # Parse frame rate which might be in format "num/den"
                if 'r_frame_rate' in stream:
                    rate = stream['r_frame_rate']
                    if '/' in rate:
                        num, den = map(int, rate.split('/'))
                        video_info['fps'] = num / den
                    else:
                        video_info['fps'] = float(rate)
                
                break
    
    return video_info

def extract_audio(filepath: str, output_path: str = None) -> str:
    """
    Extract audio from video file using ffmpeg
    
    Args:
        filepath: Path to the video file
        output_path: Path to save the extracted audio (if None, will use video filename with .wav extension)
        
    Returns:
        Path to the extracted audio file
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Video file not found: {filepath}")
    
    if output_path is None:
        output_path = filepath.with_suffix('.wav')
    else:
        output_path = Path(output_path)
    
    # Use ffmpeg to extract audio
    cmd = [
        "ffmpeg",
        "-i", str(filepath),
        "-q:a", "0",
        "-map", "a",
        "-y",  # Overwrite existing files without asking
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract audio: {result.stderr}")
    
    return str(output_path)

def get_video_frame_count(filepath: str) -> int:
    """
    Get the total number of frames in a video using OpenCV
    
    Args:
        filepath: Path to the video file
        
    Returns:
        Total frame count
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {filepath}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count
