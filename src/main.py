#!/usr/bin/env python
from pathlib import Path
import os
import json
import subprocess

from dotenv import load_dotenv
from src.downloader import download_audio
from src.transcriber import transcribe_local
from src.chunker import chunk_transcript
from src.summariser import (
    summarise_chunks, 
    format_transcript, 
    generate_master_summary, 
    generate_toc,
    extract_citations
)

def run_lms_command(args):
    """Helper to run lms CLI commands."""
    try:
        subprocess.run(["lms"] + args, check=True, capture_output=True)
    except Exception as e:
        print(f"Warning: lms command failed: {e}")

def get_unique_path(base_path: Path) -> Path:
    """If file exists, append _1, _2, etc. to the filename."""
    if not base_path.exists():
        return base_path
    
    counter = 1
    while True:
        new_path = base_path.with_name(f"{base_path.stem}_{counter}{base_path.suffix}")
        if not new_path.exists():
            return new_path
        counter += 1

def main():
    load_dotenv()  # .env variables
    model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")

    print("--- Audio Summary Pipeline ---")
    url = input("Enter YouTube URL: ").strip()
    while not url:
        url = input("URL cannot be empty. Enter YouTube URL: ").strip()

    print("\nSelect Video Type:")
    print("1. News (Multiple topics, larger overlap)")
    print("2. Topical (Single subject, larger chunks)")
    print("3. Deep Analysis (Long videos, high accuracy, citation extraction)")
    choice = input("Choice (1, 2, or 3, default is 2): ").strip()
    
    whisper_model_choice = None
    if choice == "1":
        video_type = "news"
        deep_mode = False
    elif choice == "3":
        video_type = "topical"
        deep_mode = True
        print("\nSelect Whisper Model for Deep Analysis:")
        print("1. Large-v3 (Most accurate, slower)")
        print("2. Turbo (Fast, very accurate)")
        w_choice = input("Choice (1 or 2, default is 1): ").strip()
        whisper_model_choice = "turbo" if w_choice == "2" else "large-v3"
    else:
        video_type = "topical"
        deep_mode = False
    
    print(f"\nStarting process for {video_type.upper()} video {'(DEEP MODE)' if deep_mode else ''}...")

    audio_dir = Path("data/audio")
    transcript_dir = Path("data/transcripts")
    summary_dir = Path("data/summaries")

    # Ensure directories exist
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    audio_path = download_audio(url, audio_dir)
    print(f"Downloaded: {audio_path}")

    # STEP 1: Unload LM Studio to free VRAM for Whisper
    print("Unloading LM Studio models to free VRAM...")
    run_lms_command(["unload", "--all"])

    print("Transcribing audio locally with Whisper...")
    # In deep mode, we use the user's choice or default to large-v3
    if deep_mode:
        model_to_use = whisper_model_choice
    else:
        model_to_use = os.getenv("WHISPER_MODEL", "medium")
        
    transcription_result = transcribe_local(audio_path, model_override=model_to_use)
    text = transcription_result["text"]
    segments = transcription_result["segments"]

    # STEP 2: Load LM Studio for LLM tasks
    print(f"Loading {model_name} into LM Studio (64k context)...")
    run_lms_command(["load", model_name, "--context-length", "65536", "--gpu", "max", "-y"])

    print("Generating Table of Contents...")
    toc = generate_toc(segments, api_base=os.getenv("LM_STUDIO_ENDPOINT"))
    toc_file = get_unique_path(summary_dir / f"{audio_path.stem}_toc.txt")
    toc_file.write_text(toc, encoding="utf-8")
    print(f"TOC saved: {toc_file}")

    if deep_mode:
        print("Extracting Citations & Key Terms...")
        citations = extract_citations(text, api_base=os.getenv("LM_STUDIO_ENDPOINT"))
        cite_file = get_unique_path(summary_dir / f"{audio_path.stem}_citations.txt")
        cite_file.write_text(citations, encoding="utf-8")
        print(f"Citations saved: {cite_file}")

    print("Formatting transcript into paragraphs & identifying speakers...")
    formatted_text = format_transcript(text, api_base=os.getenv("LM_STUDIO_ENDPOINT"))

    transcript_file = get_unique_path(transcript_dir / f"{audio_path.stem}.txt")
    transcript_file.write_text(formatted_text, encoding="utf-8")
    print(f"Transcript saved: {transcript_file}")

    chunks = chunk_transcript(text, video_type=video_type)
    summaries = summarise_chunks(chunks, api_base=os.getenv("LM_STUDIO_ENDPOINT"), video_type=video_type)
    
    print("Generating Master Summary...")
    master_summary = generate_master_summary(summaries, api_base=os.getenv("LM_STUDIO_ENDPOINT"))

    summary_texts = "\n\n".join(
        f"Chunk {cid}:\n{summary}" for cid, summary in summaries.items()
    )
    
    final_summary_content = f"MASTER SUMMARY:\n{master_summary}\n\n" + "="*30 + "\n\nCHUNK SUMMARIES:\n" + summary_texts
    summary_file = get_unique_path(summary_dir / f"{audio_path.stem}_summary.txt")
    summary_file.write_text(final_summary_content, encoding="utf-8")
    print(f"Summary saved: {summary_file}")

    # STEP 3: Final Unload to leave GPU clean
    print("Final cleanup: Unloading LM Studio models...")
    run_lms_command(["unload", "--all"])

if __name__ == "__main__":
    main()