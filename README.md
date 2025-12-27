# AudioSummary: Local YouTube Intelligence Pipeline

A high-performance, fully local pipeline for downloading, transcribing, and analyzing YouTube content. Optimized for NVIDIA RTX 50-series (Blackwell) hardware and long-form content (80+ minutes).

## Features

- **100% Local**: No API keys or cloud costs. Uses local Whisper and LM Studio.
- **Deep Analysis Mode**: Generates a Master Summary, Table of Contents, and Citation Extraction.
- **VRAM Management**: Automatically loads/unloads models using the `lms` CLI to fit within 16GB VRAM (RTX 5090 Mobile).
- **High Accuracy**: Supports Whisper `large-v3` and `turbo` models.
- **Long-Form Optimized**: Configured with a 64K context window to handle transcripts from videos over 2 hours long.
- **Smart Versioning**: Automatically handles duplicate filenames by adding version suffixes.

## Prerequisites

1.  **Python 3.12+**
2.  **FFmpeg**: Must be installed and added to your system PATH.
3.  **LM Studio**: Must have the `lms` CLI installed and a model downloaded (e.g., `Llama-3.1-8B-Instruct`).
4.  **NVIDIA GPU**: Optimized for RTX 5090 (requires PyTorch Nightly).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/tourniquetrules/audiosummary.git
    cd audiosummary
    ```

2.  **Create and activate a virtual environment**:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install PyTorch Nightly (for RTX 5090/Blackwell support)**:
    ```powershell
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
    ```

4.  **Install dependencies**:
    ```powershell
    pip install yt-dlp openai tiktoken pydantic tqdm python-dotenv openai-whisper
    ```

## Configuration

Create a `.env` file in the root directory:

```dotenv
OPENAI_API_KEY=lm-studio
LM_STUDIO_ENDPOINT=http://localhost:1234/v1
# The model identifier in LM Studio
MODEL_NAME=lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
WHISPER_MODEL=large-v3
SUMMARY_TEMPERATURE=0.3
```

## Usage

The easiest way to run the project is via the provided batch file:

```powershell
.\run.bat
```

### Interactive CLI Options:
- **News Mode**: Summarizes multiple topics within a single video.
- **Topical Mode**: Focuses on one main subject.
- **Deep Analysis**: 
    - Uses Whisper `large-v3` or `turbo`.
    - Generates a Master Summary.
    - Creates a Table of Contents.
    - Extracts key citations.
    - Automatically manages VRAM by unloading models during transcription.

## Project Structure

- `src/main.py`: Orchestrator and CLI logic.
- `src/downloader.py`: YouTube audio extraction using `yt-dlp`.
- `src/transcriber.py`: Local Whisper transcription with CUDA acceleration.
- `src/summariser.py`: LLM analysis logic (Master Summary, TOC, Citations).
- `src/chunker.py`: Intelligent transcript splitting for large context windows.
- `data/`: Output directory for transcripts and summaries.
