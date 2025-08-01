# filepath: /home/Maanu/Documents/video-embeddings/utils/check_processed_utils.py
"""
check_processed_utils.py

Utility functions to track which videos have been processed
to avoid reprocessing them when running the pipeline again.
"""

import os
import json
import time
from pathlib import Path

# Path to the processed videos log file
PROCESSED_LOG_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data" / "processed_videos.json"

def ensure_log_exists():
    """
    Ensures the processed videos log file exists
    """
    os.makedirs(os.path.dirname(PROCESSED_LOG_PATH), exist_ok=True)
    
    if not os.path.exists(PROCESSED_LOG_PATH):
        # Create initial empty log file
        with open(PROCESSED_LOG_PATH, 'w') as f:
            json.dump({"processed_videos": []}, f, indent=2)

def load_processed_log():
    """
    Load the processed videos log
    
    Returns:
        dict: The processed videos log data
    """
    ensure_log_exists()
    
    try:
        with open(PROCESSED_LOG_PATH, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("Warning: Error reading processed videos log. Creating a new one.")
        log_data = {"processed_videos": []}
        save_processed_log(log_data)
        return log_data

def save_processed_log(log_data):
    """
    Save the processed videos log
    
    Args:
        log_data (dict): The log data to save
    """
    ensure_log_exists()
    
    with open(PROCESSED_LOG_PATH, 'w') as f:
        json.dump(log_data, f, indent=2)

def is_video_processed(video_path):
    """
    Check if a video has already been processed
    
    Args:
        video_path (str or Path): Path to the video file
    
    Returns:
        bool: True if the video has already been processed, False otherwise
    """
    # Convert to string and get absolute path for consistency
    video_path = str(Path(video_path).resolve())
    
    log_data = load_processed_log()
    
    # Check if video path is in the processed list
    for entry in log_data["processed_videos"]:
        if entry["video_path"] == video_path:
            return True
    
    return False

def mark_video_processed(video_path, metadata=None):
    """
    Mark a video as processed in the log
    
    Args:
        video_path (str or Path): Path to the video file
        metadata (dict, optional): Additional metadata about the processing
    """
    # Convert to string and get absolute path for consistency
    video_path = str(Path(video_path).resolve())
    video_name = Path(video_path).name
    
    log_data = load_processed_log()
    
    # Prepare the entry
    entry = {
        "video_path": video_path,
        "video_name": video_name,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Add any additional metadata
    if metadata:
        entry.update(metadata)
    
    # Remove any existing entry for this video
    log_data["processed_videos"] = [v for v in log_data["processed_videos"] if v["video_path"] != video_path]
    
    # Add the new entry
    log_data["processed_videos"].append(entry)
    
    # Save the updated log
    save_processed_log(log_data)
    
    print(f"Video {video_name} marked as processed in the log")

def get_unprocessed_videos(directory):
    """
    Get a list of videos in a directory that have not yet been processed
    
    Args:
        directory (str or Path): Directory to check for videos
    
    Returns:
        list: List of unprocessed video file paths
    """
    directory = Path(directory)
    
    # Get all video files in the directory
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    all_videos = [str(f.resolve()) for f in directory.glob("*") 
                 if f.suffix.lower() in video_extensions]
    
    # Filter out already processed videos
    unprocessed_videos = [video for video in all_videos if not is_video_processed(video)]
    
    return unprocessed_videos