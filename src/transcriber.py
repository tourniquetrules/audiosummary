from pathlib import Path
import os

def transcribe_local(audio_path: Path, model_override=None) -> dict:
    """
    Transcribe audio using local whisper library.
    Returns a dictionary with 'text' and 'segments'.
    """
    try:
        import whisper
        import torch
    except ImportError:
        raise ImportError("Please install 'openai-whisper' and 'torch' to use local transcription.")
    
    model_name = model_override if model_override else os.getenv("WHISPER_MODEL", "medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Model: {model_name}")
    
    model = whisper.load_model(model_name, device=device)
    
    # fp16 is only supported on CUDA
    result = model.transcribe(str(audio_path), fp16=(device == "cuda"))
    
    # Explicitly delete model and empty cache to free VRAM for LM Studio
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    return {
        "text": result["text"],
        "segments": result["segments"]
    }
