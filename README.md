# Time-Segmented Multimodal Video Embedding System

A powerful system for transcribing videos, segmenting them by sentences, generating multimodal embeddings, and enabling semantic search through video content. This system allows you to build a searchable video library with advanced semantic understanding.

## 🎯 Project Goal

Build a system that:

- Transcribes a video into sentence-level text using Whisper
- Segments the video based on sentence timestamps (not fixed time chunks)
- For each sentence segment:
  - Extracts the relevant frame and audio
  - Creates a multimodal embedding (visual + textual) using Google's Embedding API
- Stores these embeddings along with rich metadata in Pinecone
- Allows users to search semantically using natural language queries
- Refines results using LLM (Gemini Pro) for better, contextual understanding
- Tracks processed videos to avoid redundant processing

## 💡 Why This Approach?

- Traditional fixed-time chunking can cut sentences in half, breaking semantic meaning
- Segmenting by sentence boundaries from the transcript gives better alignment between:
  - What was said
  - What was seen/heard
- Multimodal embeddings (frame + audio/transcript) allow richer semantic understanding
- Using LLM post-retrieval improves result quality via reasoning

## 🧩 System Architecture

```
Video
│
├──► Transcribe with Whisper → Sentence timestamps
│
├──► Segment video/audio per sentence
│
├──► Generate multimodal embedding (Gemini Pro)
│
├──► Store in Pinecone (vector + metadata)
│
└──► Semantic search → Top-k → LLM refinement
```

## 📋 Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- API keys for:
  - Google AI (for Gemini API and text embeddings)
  - Pinecone (for vector storage)

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/video-embeddings.git
cd video-embeddings
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
GEMINI_API_KEY=your_google_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_pinecone_index_name
```

## 📁 Project Structure

```
video-embeddings/
├── config/
│   └── settings.py           # Configuration and environment variables
├── data/
│   ├── raw_videos/           # Input videos
│   ├── processed/            # Processed transcripts
│   ├── chunks/               # Video/audio chunks and frames
│   ├── embeddings/           # Generated embeddings
│   └── processed_videos.json # Tracking log for processed videos
├── pipeline/
│   ├── run_transcription.py  # Whisper transcription
│   ├── generate_chunks.py    # Sentence-level video chunking
│   ├── generate_embeddings.py # Create multimodal embeddings
│   └── upload_to_pinecone.py # Upload to vector database
├── utils/
│   ├── transcript_utils.py   # Whisper and transcript processing
│   ├── video_utils.py        # Video metadata and processing
│   ├── chunk_utils.py        # Extract frames and audio segments
│   ├── embedding_utils.py    # Generate embeddings with Gemini
│   └── check_processed_utils.py # Track processed videos
├── main.py                   # Run the full pipeline
├── search_video.py           # Search for video segments
└── requirements.txt
```

## 🔧 Usage

### Process a video

```bash
# Process a single video
python main.py /path/to/your/video.mp4

# Process all unprocessed videos in data/raw_videos/
python main.py

# Process all videos including already processed ones
python main.py --process-all

# Force reprocessing of all videos
python main.py --force

# Skip uploading to Pinecone
python main.py --skip-upload

# Skip specific pipeline steps
python main.py --skip-transcription
python main.py --skip-chunks
python main.py --skip-embeddings
```

### Search for video content

```bash
# Basic search
python search_video.py "your search query"

# Limit results
python search_video.py "your search query" --top-k 3

# Search within a specific video
python search_video.py "your search query" --video-id your_video_id

# Refine results with LLM
python search_video.py "your search query" --refine
```

### Run individual pipeline components

```bash
# Only transcribe
python pipeline/run_transcription.py /path/to/your/video.mp4

# Only generate chunks
python pipeline/generate_chunks.py /path/to/processed_transcript.json

# Only generate embeddings
python pipeline/generate_embeddings.py /path/to/chunks_summary.json

# Only upload to Pinecone
python pipeline/upload_to_pinecone.py /path/to/embeddings.json
```

## 📝 Notes

- For best results, use high-quality videos with clear audio
- The Whisper model size can be configured in settings.py (tiny, base, small, medium, large)
- Gemini Pro requires clear frames for optimal multimodal understanding
- The system automatically handles segments with no speech

## 📊 Performance Considerations

- Transcription is CPU-intensive and might be slow on large videos
- Embedding generation makes API calls to Google AI and may incur costs
- Consider batch processing for large video collections

## ⚠️ Important Notes

- The system tracks which videos have been processed to avoid redundant processing
- The tracking information is stored in `data/processed_videos.json`
- By default, only new (unprocessed) videos will be processed when running `python main.py`
- Use the `--process-all` flag to process all videos regardless of their processed status
- Use the `--force` flag to fully reprocess videos (ignoring existing processed data)
- When working with a team, consider excluding `data/processed_videos.json` from version control if your processing state should be local

## 📄 Logging and Tracking

The system maintains a log of processed videos at `data/processed_videos.json` with:

- Video path
- Video name
- Processing timestamp
- Additional metadata

This tracking system enables:

1. Skipping already processed videos (default behavior)
2. Only processing newly added videos in a directory
3. Forced reprocessing when needed with the `--force` flag
4. Processing all videos regardless of status with `--process-all`
