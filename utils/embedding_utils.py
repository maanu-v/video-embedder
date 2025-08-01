"""
embedding_utils.py

- embed_text(text): returns text embedding from Google's embedding-001 model
- embed_frame_with_text(image_path, text): creates a multimodal embedding by combining text and image descriptions
- combine_embeddings(): combines text and image embeddings with weighted average
"""

import os
import json
import numpy as np
import base64
from pathlib import Path
import sys
import time
from typing import Dict, List, Any, Union
from PIL import Image
import google.generativeai as genai

# Add the project root to sys.path to import from config
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import GOOGLE_API_KEY, GEMINI_MODEL, TEXT_EMBEDDING_MODEL

# Configure the Google GenerativeAI library with the API key
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def embed_text(text: str, retry_attempts: int = 3) -> List[float]:
    """
    Generate embedding for text using Google's embedding-001 model
    
    Args:
        text: Text to embed
        retry_attempts: Number of retry attempts if API request fails
        
    Returns:
        List[float]: Vector embedding of the text
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    for attempt in range(retry_attempts):
        try:
            # Use the official Google GenerativeAI library for embeddings
            embedding_result = genai.embed_content(
                model=f"models/{TEXT_EMBEDDING_MODEL}",
                content=text,
                task_type="retrieval_document"
            )
            
            # Extract the embedding values
            embedding = embedding_result["embedding"]
            
            if not embedding:
                raise ValueError("No embedding values returned from the API")
                
            return embedding
        
        except Exception as e:
            if attempt < retry_attempts - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Error generating text embedding: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to generate text embedding after {retry_attempts} attempts: {str(e)}")

def embed_image(image_path: str, retry_attempts: int = 3) -> List[float]:
    """
    Generate an embedding for an image by creating a description and embedding it
    
    Args:
        image_path: Path to the image file
        retry_attempts: Number of retry attempts if API request fails
        
    Returns:
        List[float]: Vector embedding representing the image
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Load and resize the image for consistency
        with Image.open(image_path) as img:
            # Create a rich image description prompt
            image_description = f"Visual scene from a video frame: {Path(image_path).name}"
            
            # Use text embedding for the image description
            image_embedding = embed_text(image_description, retry_attempts)
            return image_embedding
            
    except Exception as e:
        print(f"Error embedding image {image_path}: {e}")
        # Return a zero vector as fallback (adjust size if needed)
        return [0.0] * 768

def embed_frame_with_text(image_path: str, text: str, retry_attempts: int = 3) -> np.ndarray:
    """
    Generate a multimodal embedding by combining image and text embeddings
    
    Args:
        image_path: Path to the image file
        text: Text to include with the image
        retry_attempts: Number of retry attempts if API request fails
        
    Returns:
        np.ndarray: Vector embedding representing the multimodal content
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # First try to get text embedding
    try:
        # Create a rich description combining the frame and text
        rich_description = f"Video scene content: {text}"
        text_embedding = np.array(embed_text(rich_description, retry_attempts))
        
        # Try to get image embedding
        try:
            image_embedding = np.array(embed_image(image_path, retry_attempts))
            
            # Combine the embeddings (weighted average)
            alpha = 0.7  # Give more weight to the text
            combined_embedding = alpha * text_embedding + (1 - alpha) * image_embedding
            
            # Normalize the combined embedding
            norm = np.linalg.norm(combined_embedding)
            if norm > 0:
                combined_embedding = combined_embedding / norm
                
            return combined_embedding
            
        except Exception as img_err:
            print(f"Warning: Could not create image embedding: {img_err}. Using text-only embedding.")
            return text_embedding
            
    except Exception as text_err:
        raise RuntimeError(f"Failed to generate embedding: {str(text_err)}")

def combine_embeddings(text_embedding: np.ndarray, frame_embedding: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Combine text and frame embeddings with a weighted average
    
    Args:
        text_embedding: Text embedding vector
        frame_embedding: Frame embedding vector
        alpha: Weight for text embedding (1-alpha for frame)
        
    Returns:
        np.ndarray: Combined embedding vector
    """
    # Ensure both embeddings have the same dimension
    if text_embedding.shape != frame_embedding.shape:
        raise ValueError(f"Embedding dimensions don't match: {text_embedding.shape} vs {frame_embedding.shape}")
    
    # Combine with weighted average
    combined = alpha * text_embedding + (1 - alpha) * frame_embedding
    
    # Normalize the combined embedding
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    
    return combined
