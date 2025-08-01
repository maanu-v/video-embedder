# filepath: /home/Maanu/Documents/video-embeddings/search_video.py
"""
search_video.py

Allows users to:
1. Search videos using text queries
2. Use LLM (Gemini) to refine and contextualize results
3. Return relevant video segments with timestamps
"""

import os
import json
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import requests
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    GOOGLE_API_KEY,
    TEXT_EMBEDDING_MODEL
)

# Import pinecone and configure
import pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Import Google Generative AI library if available
try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google.generativeai library not available. Install with: pip install google-generativeai")

def embed_text(text: str) -> List[float]:
    """
    Generate embedding for text using Google's embedding-001 model
    
    Args:
        text: Text to embed
        
    Returns:
        List[float]: Vector embedding of the text
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    if not GENAI_AVAILABLE:
        raise ImportError("google.generativeai library not installed")
    
    try:
        # Use the official Google GenerativeAI library for embeddings
        embedding_result = genai.embed_content(
            model=f"models/{TEXT_EMBEDDING_MODEL}",
            content=text,
            task_type="retrieval_query"  # Use retrieval_query for search queries
        )
        
        # Extract the embedding values
        embedding = embedding_result["embedding"]
        
        if not embedding:
            raise ValueError("No embedding values returned from the API")
            
        return embedding
    
    except Exception as e:
        print(f"Error generating text embedding: {str(e)}")
        raise RuntimeError(f"Failed to generate text embedding: {str(e)}")

def search_videos(query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
    """
    Search videos using a text query
    
    Args:
        query: Text query to search for
        top_k: Number of results to return
        filter_dict: Optional filter dictionary for Pinecone query
    
    Returns:
        List of search results
    """
    # Get text embedding for query
    query_embedding = embed_text(query)
    
    # Connect to Pinecone index
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Search Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )
    
    # Format results
    formatted_results = []
    for match in search_results.matches:
        result = {
            'score': match.score,
            'chunk_id': match.id,
            'metadata': match.metadata
        }
        formatted_results.append(result)
    
    return formatted_results

def refine_results_with_gemini(query: str, results: List[Dict]) -> str:
    """
    Use Gemini to refine and contextualize search results
    
    Args:
        query: Original user query
        results: Search results from Pinecone
        
    Returns:
        Refined analysis as a string
    """
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not set. Skipping LLM refinement.")
        return "No API key available for refinement."
    
    if not results:
        print("No results to refine.")
        return "No results to analyze."
    
    # Prepare prompt for Gemini
    contexts = []
    for i, result in enumerate(results):
        metadata = result['metadata']
        context = f"Segment {i+1}:\n"
        context += f"Text: {metadata.get('text', 'No text available')}\n"
        context += f"Time: {float(metadata.get('start_time', 0)):.2f}s - {float(metadata.get('end_time', 0)):.2f}s\n"
        context += f"Video ID: {metadata.get('video_id', 'Unknown')}\n"
        context += f"Score: {result['score']:.4f}\n"
        contexts.append(context)
    
    context_text = "\n".join(contexts)
    
    prompt = f"""
    Given a query and search results from a video, analyze which segments best answer the query.
    Re-rank the segments based on relevance to the query.
    
    Query: {query}
    
    Search results:
    {context_text}
    
    Please provide:
    1. A brief explanation of which segments are most relevant and why
    2. The segments in ranked order (most relevant first)
    3. For each segment, include the segment number and a brief explanation of its relevance
    """
    
    if GENAI_AVAILABLE:
        try:
            # Use the Google GenerativeAI library for content generation
            # Use gemini-pro instead of gemini-pro-vision
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            explanation = response.text
            return explanation
        except Exception as e:
            print(f"Error using Google Generative AI library: {str(e)}")
            # Fall back to REST API
    
    # Fall back to REST API if needed
    try:
        # Call Gemini API using REST
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GOOGLE_API_KEY
        }
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 1024
            }
        }
        
        # Use gemini-pro instead of gemini-pro-vision
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        llm_response = response.json()
        explanation = llm_response['candidates'][0]['content']['parts'][0]['text']
        return explanation
        
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Failed to get refinement: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Search through video embeddings")
    parser.add_argument("query", help="Text query to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--video-id", help="Filter results to specific video ID")
    parser.add_argument("--refine", action="store_true", help="Use Gemini to refine results")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()
    
    # Prepare filter if video_id is specified
    filter_dict = None
    if args.video_id:
        filter_dict = {"video_id": {"$eq": args.video_id}}
    
    # Search videos
    print(f"Searching for: '{args.query}'")
    try:
        results = search_videos(args.query, args.top_k, filter_dict)
        
        if not results:
            print("No results found.")
            return
        
        # Print results
        print("\nSearch Results:")
        print("==============")
        
        for i, result in enumerate(results):
            metadata = result['metadata']
            
            # Extract video name from video_path (if available)
            video_name = "Unknown"
            if metadata.get('video_path'):
                video_path = metadata.get('video_path')
                # Extract file name without extension
                video_name = os.path.basename(video_path).split('.')[0]
            elif metadata.get('video_id'):
                video_name = metadata.get('video_id')
                
            # Get chunk information
            chunk_id = result.get('chunk_id', 'Unknown')
            # chunk_index = metadata.get('chunk_index', 'N/A')
            
            print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
            print(f"Video: {video_name}")
            print(f"Chunk: {chunk_id} ")
            print(f"Timestamp: {float(metadata.get('start_time', 0)):.2f}s - {float(metadata.get('end_time', 0)):.2f}s")
            print(f"Text: \"{metadata.get('text', 'No text available')}\"")
        
        # Refine results if requested
        if args.refine:
            print("\nRefining results with Gemini...")
            analysis = refine_results_with_gemini(args.query, results)
            
            print("\nGemini Analysis:")
            print("===============")
            print(analysis)
            
        # Save results to file if requested
        if args.output:
            output_data = {
                "query": args.query,
                "results": [
                    {
                        "score": r["score"],
                        "chunk_id": r["chunk_id"],
                        "video_id": r["metadata"].get("video_id", "Unknown"),
                        "start_time": float(r["metadata"].get("start_time", 0)),
                        "end_time": float(r["metadata"].get("end_time", 0)),
                        "text": r["metadata"].get("text", "No text available")
                    }
                    for r in results
                ]
            }
            
            if args.refine:
                output_data["gemini_analysis"] = analysis
                
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()