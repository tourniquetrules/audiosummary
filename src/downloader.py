# src/downloader.py
import os
from pathlib import Path
import yt_dlp

def download_audio(url: str, output_dir: Path) -> Path:
    """
    Download the best‑quality audio stream from YouTube and return its file path.

    Parameters
    ----------
    url : str
        Full YouTube URL.
    output_dir : pathlib.Path
        Directory where the audio will be stored (will be created if missing).

    Returns
    -------
    pathlib.Path
        Path to the downloaded .mp3 file.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",                     # best audio quality
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"},
        ],
        "quiet": True,                                   # silence yt-dlp output
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Get the actual filename yt-dlp generated
        temp_filename = ydl.prepare_filename(info)
        # Replace the original extension with .mp3 as per postprocessor
        audio_path = Path(temp_filename).with_suffix(".mp3")

    return audio_path
