"""
transcript_utils.py

- transcribe_video(filepath): runs Whisper on full video
- split_into_sentences(transcript): splits based on punctuation
- map_sentences_to_timestamps(words, sentences): uses Whisper word timestamps to assign start/end per sentence
"""

import os
import json
import whisper
import nltk
from typing import List, Dict, Any, Tuple
import re
import numpy as np
from pathlib import Path
import sys

# Add the project root to sys.path to import from config
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import WHISPER_MODEL, LANGUAGE

# Download the necessary NLTK resources for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def transcribe_video(filepath: str, output_path: str = None) -> Dict[str, Any]:
    """
    Transcribe a video file using OpenAI's Whisper model
    
    Args:
        filepath: Path to the video file
        output_path: Path to save the transcription result (if None, will use video filename with _transcript.json)
        
    Returns:
        Dictionary containing the transcription result
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Video file not found: {filepath}")
    
    if output_path is None:
        output_path = filepath.parent / f"{filepath.stem}_transcript.json"
    else:
        output_path = Path(output_path)
    
    # Load the Whisper model
    model = whisper.load_model(WHISPER_MODEL)
    
    # Transcribe with word-level timestamps
    result = model.transcribe(
        str(filepath),
        language=LANGUAGE,
        word_timestamps=True,
        verbose=True
    )
    
    # Save the result to a JSON file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def split_into_sentences(transcript: str) -> List[str]:
    """
    Split a transcript into sentences using NLTK's punkt tokenizer
    
    Args:
        transcript: Full transcript text
        
    Returns:
        List of sentences
    """
    # Use NLTK's sentence tokenizer
    sentences = nltk.sent_tokenize(transcript)
    
    # Clean up sentences (remove extra whitespace, etc.)
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s]  # Remove empty sentences
    
    return sentences

def map_sentences_to_timestamps(
    words: List[Dict[str, Any]], 
    sentences: List[str]
) -> List[Dict[str, Any]]:
    """
    Map sentences to their start and end timestamps using word-level timestamps from Whisper
    
    Args:
        words: List of words with timestamps from Whisper
        sentences: List of sentences from the transcript
        
    Returns:
        List of dictionaries containing sentences with their start and end timestamps
    """
    # Convert words to a format easier to work with
    word_data = []
    for word_info in words:
        word_data.append({
            'word': word_info['word'],
            'start': word_info['start'],
            'end': word_info['end']
        })
    
    # Create a single string with all words
    full_text = ' '.join([w['word'] for w in word_data])
    
    sentence_data = []
    current_pos = 0
    
    for sentence in sentences:
        # Clean up the sentence to match how it might appear in the full text
        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        # Find the position of this sentence in the full text, starting from current_pos
        sentence_pos = full_text.find(clean_sentence, current_pos)
        
        if sentence_pos == -1:
            # If exact match not found, try a more flexible approach
            # This is needed because NLTK's sentence tokenization might differ slightly from the raw words
            best_match_pos = -1
            best_match_score = -1
            
            # Look for the first few words of the sentence
            first_words = ' '.join(clean_sentence.split()[:3])
            if first_words:
                fw_pos = full_text.find(first_words, current_pos)
                if fw_pos != -1:
                    sentence_pos = fw_pos
        
        if sentence_pos == -1:
            # If still not found, skip this sentence
            continue
        
        # Update the current position for the next search
        current_pos = sentence_pos + len(clean_sentence)
        
        # Count words before the sentence
        words_before = full_text[:sentence_pos].count(' ')
        
        # Count words in the sentence
        words_in_sentence = clean_sentence.count(' ') + 1
        
        # Find the start and end timestamps
        start_idx = words_before
        end_idx = words_before + words_in_sentence - 1
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(word_data) - 1))
        end_idx = max(0, min(end_idx, len(word_data) - 1))
        
        # Get the timestamps
        start_time = word_data[start_idx]['start']
        end_time = word_data[end_idx]['end']
        
        sentence_data.append({
            'sentence': sentence,
            'start': start_time,
            'end': end_time
        })
    
    return sentence_data

def process_transcript(transcript_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a Whisper transcription result to get sentences with timestamps
    
    Args:
        transcript_result: Whisper transcription result
        
    Returns:
        List of dictionaries containing sentences with their start and end timestamps
    """
    # Extract segments directly from Whisper
    segments = transcript_result.get('segments', [])
    
    # Process each segment to get sentence-level data
    sentence_data = []
    
    for segment in segments:
        # Get segment text and timestamps
        text = segment.get('text', '').strip()
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        
        # Skip empty segments
        if not text:
            continue
            
        # Split segment into sentences if it contains multiple sentences
        sentences = split_into_sentences(text)
        
        # For each sentence, try to estimate its position within the segment
        if len(sentences) == 1:
            # Just one sentence - use the segment's timestamps
            sentence_data.append({
                'sentence': sentences[0],
                'start': start,
                'end': end
            })
        else:
            # Multiple sentences - distribute the time proportionally based on character count
            total_chars = sum(len(s) for s in sentences)
            segment_duration = end - start
            
            current_start = start
            for sentence in sentences:
                # Calculate approximate duration based on character count
                sentence_portion = len(sentence) / total_chars
                sentence_duration = segment_duration * sentence_portion
                sentence_end = current_start + sentence_duration
                
                sentence_data.append({
                    'sentence': sentence,
                    'start': current_start,
                    'end': sentence_end
                })
                
                current_start = sentence_end
    
    # Sort by start time
    sentence_data.sort(key=lambda x: x['start'])
    
    return sentence_data

def handle_non_transcribed_segments(
    sentence_data: List[Dict[str, Any]], 
    video_duration: float,
    min_segment_length: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Identify gaps in the transcription and create "silent" segments for them
    
    Args:
        sentence_data: List of sentence data with timestamps
        video_duration: Total duration of the video in seconds
        min_segment_length: Minimum length in seconds for a segment to be considered
        
    Returns:
        Updated list of sentence data including silent segments
    """
    if not sentence_data:
        # If no transcription at all, create a single segment for the whole video
        return [{
            'sentence': '[NO SPEECH DETECTED]',
            'start': 0.0,
            'end': video_duration,
            'is_silent': True
        }]
    
    # Sort segments by start time
    sentence_data = sorted(sentence_data, key=lambda x: x['start'])
    
    # Add is_silent flag to existing segments (they're not silent)
    for segment in sentence_data:
        segment['is_silent'] = False
    
    result = []
    current_time = 0.0
    
    # Check for gap at the beginning
    if sentence_data[0]['start'] > min_segment_length:
        result.append({
            'sentence': '[NO SPEECH DETECTED]',
            'start': 0.0,
            'end': sentence_data[0]['start'],
            'is_silent': True
        })
    
    # Add first segment
    result.append(sentence_data[0])
    current_time = sentence_data[0]['end']
    
    # Check for gaps between segments
    for i in range(1, len(sentence_data)):
        gap = sentence_data[i]['start'] - current_time
        if gap > min_segment_length:
            result.append({
                'sentence': '[NO SPEECH DETECTED]',
                'start': current_time,
                'end': sentence_data[i]['start'],
                'is_silent': True
            })
        
        result.append(sentence_data[i])
        current_time = sentence_data[i]['end']
    
    # Check for gap at the end
    if video_duration - current_time > min_segment_length:
        result.append({
            'sentence': '[NO SPEECH DETECTED]',
            'start': current_time,
            'end': video_duration,
            'is_silent': True
        })
    
    return result
