# src/summariser.py
import os
import openai
from typing import List, Dict

def summarise_chunks(chunks: List[Dict], api_base=None, video_type="topical") -> Dict[str, str]:
    """
    Call LM Studio or OpenAI ChatCompletion to summarize each chunk.

    Parameters
    ----------
    chunks : List[Dict]
        [{'id': '1', 'text': '...'}, ...]
    api_base : str | None
        Base URL for LM Studio.
    video_type : str
        'news' or 'topical'. Changes the summarization prompt.
    """
    client = openai.OpenAI(base_url=api_base) if api_base else openai.OpenAI()
    summaries = {}
    
    for chunk in chunks:
        if video_type == "news":
            prompt = (
                "You are a news editor. Summarize this transcript segment. "
                "Identify each distinct news story or topic mentioned. "
                "Use bullet points for each new topic. Segment:\n\n"
                f"{chunk['text']}"
            )
        else:
            prompt = (
                "You are an expert analyst. Summarize the main arguments and key points "
                "of this discussion. Maintain the logical flow of the argument. "
                "Use bullet points. Segment:\n\n"
                f"{chunk['text']}"
            )

        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=float(os.getenv("SUMMARY_TEMPERATURE", 0.3)),
            max_tokens=256,
        )
        summaries[chunk["id"]] = response.choices[0].message.content.strip()
    return summaries

def format_transcript(text: str, api_base=None) -> str:
    "Use the LLM to format a raw transcript into logical paragraphs and identify speakers in chunks."
    client = openai.OpenAI(base_url=api_base) if api_base else openai.OpenAI()
    
    # Increased chunk size to ~15,000 characters (~4000 tokens) for better coherence with 64k context
    chunk_size = 15000
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    formatted_parts = []

    print(f"Formatting transcript in {len(text_chunks)} segments...")
    
    for i, chunk in enumerate(text_chunks):
        prompt = (
            "The following is a segment of a raw transcript. It lacks paragraph breaks and speaker labels. "
            "Please format it into logical paragraphs to make it readable. "
            "Identify when the speaker changes and label them (e.g., 'Speaker 1:', 'Speaker 2:') based on context. "
            "DO NOT change any words, DO NOT add a summary, and DO NOT add your own commentary. "
            f"Segment {i+1}/{len(text_chunks)}:\n\n"
            f"{chunk}"
        )

        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        formatted_parts.append(response.choices[0].message.content.strip())
    
    return "\n\n".join(formatted_parts)


def generate_master_summary(summaries: Dict[str, str], api_base=None) -> str:
    "Combine all chunk summaries into one cohesive master summary."
    client = openai.OpenAI(base_url=api_base) if api_base else openai.OpenAI()
    combined_text = "\n\n".join(summaries.values())
    
    prompt = (
        "The following are summaries of different parts of a video. "
        "Please synthesize them into one cohesive, high-level executive summary. "
        "Highlight the main themes, the overall conclusion, and any key takeaways. "
        "Keep it professional and well-structured.\n\n"
        f"Summaries:\n{combined_text}"
    )

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def generate_toc(segments: List[Dict], api_base=None) -> str:
    "Generate a timestamped Table of Contents from transcript segments."
    client = openai.OpenAI(base_url=api_base) if api_base else openai.OpenAI()
    
    sampled_segments = []
    last_time = -300
    for seg in segments:
        if seg['start'] >= last_time + 300:
            timestamp = f"[{int(seg['start'] // 60):02d}:{int(seg['start'] % 60):02d}]"
            sampled_segments.append(f"{timestamp} {seg['text']}")
            last_time = seg['start']

    segments_text = "\n".join(sampled_segments)
    
    prompt = (
        "Based on the following timestamped transcript snippets, create a concise Table of Contents "
        "for the video. Each entry should have a timestamp and a brief description of the topic being discussed. "
        "Focus on major topic shifts.\n\n"
        f"Snippets:\n{segments_text}"
    )

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def extract_citations(text: str, api_base=None) -> str:
    "Extract Bible verses, authors, and key theological terms from the transcript in segments."
    client = openai.OpenAI(base_url=api_base) if api_base else openai.OpenAI()
    
    # Increased sample size to 15,000 characters for better coverage with 64k context
    sample_size = 15000
    samples = []
    if len(text) <= sample_size * 3:
        samples.append(text)
    else:
        samples.append(text[:sample_size]) # Start
        samples.append(text[len(text)//2 : len(text)//2 + sample_size]) # Middle
        samples.append(text[-sample_size:]) # End

    combined_samples = "\n--- SEGMENT ---\n".join(samples)
    
    prompt = (
        "The following are segments from a theological transcript. "
        "Please extract a comprehensive list of all Bible verses cited, authors mentioned, "
        "and key technical or theological terms used. "
        "Format this as a clean, categorized list.\n\n"
        f"Transcript Segments:\n{combined_samples}"
    )

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()
