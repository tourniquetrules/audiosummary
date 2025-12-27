You are an expert python programmer.  You are developing a method to download youtube videos, transcribe them and then summarize them using a language model.
You need to install the appropriate software including yt-dlp.
The expected flow is the user runs the program and enters a youtube URL.
The program then downloads the audio portion of the video and then transcribes it locally through whisper.
The resulting transcript is saved in a separate directory with a logical naming structure.
The transcript will be either a news based video that has numerous topics contained within it or a topical video with one main topic.
Allow overlap when chunking the transcript to get accurate summaries.
The final goal is to produce summaries of the transcript that the user can read at their leisure using a local LM Studio instance.
The LM Studio endpoint is http://localhost:1234/v1 and the language model used is openai/gpt-oss-20b

## Setup and Usage

1. **Install Dependencies**:
   ```bash
   pip install yt-dlp openai tiktoken pydantic tqdm python-dotenv openai-whisper
   ```
   **For GPU Acceleration (NVIDIA RTX 50-series / Blackwell)**:
   If you have an NVIDIA RTX 50-series GPU (e.g., RTX 5090), you need the nightly version of PyTorch for Blackwell support:
   ```bash
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
   ```

   **For other NVIDIA GPUs**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
   ```

2. **System Requirements**:
   - **FFmpeg**: Required by `yt-dlp` for audio conversion and `whisper` for processing. Ensure it is installed and in your PATH.

3. **Configuration**:
   Create a `.env` file in the root directory with the following:
   ```dotenv
   OPENAI_API_KEY=lm-studio
   LM_STUDIO_ENDPOINT=http://localhost:1234/v1
   MODEL_NAME=openai/gpt-oss-20b
   WHISPER_MODEL=medium
   SUMMARY_TEMPERATURE=0.3
   ```

4. **Run the Program**:
   ```bash
   python -m src.main "https://www.youtube.com/watch?v=VIDEO_ID"
   ```
