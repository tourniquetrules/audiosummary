# src/chunker.py
import re
from pathlib import Path
from typing import List, Dict

def chunk_transcript(text: str, max_tokens=1500, overlap=200, video_type="topical") -> List[Dict]:
    """
    Split transcript into overlapping chunks.

    Parameters
    ----------
    text : str
        Full transcript.
    max_tokens : int
        Rough token limit per chunk.
    overlap : int
        Number of tokens to overlap between chunks.
    video_type : str
        'news' or 'topical'. Adjusts chunking strategy.
    """
    if video_type == "news":
        # News segments benefit from smaller chunks and larger overlap 
        # to avoid missing quick topic transitions.
        max_tokens = 1000
        overlap = 300
    else:
        # Topical videos benefit from larger chunks to keep the argument flow.
        max_tokens = 2000
        overlap = 200

    # Simple sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []
    token_count, id_counter = 0, 1

    for sent in sentences:
        tokens = len(sent.split())  # crude token count
        if token_count + tokens > max_tokens and current:
            chunks.append({"id": str(id_counter), "text": " ".join(current)})
            id_counter += 1
            # overlap: keep last `overlap` tokens of previous chunk
            overlap_text = " ".join(current[-overlap:])
            current = overlap_text.split()
            token_count = len(overlap_text.split())
        current.append(sent)
        token_count += tokens

    if current:
        chunks.append({"id": str(id_counter), "text": " ".join(current)})

    return chunks