# Real-time Voice Translation Pipeline

A high-performance, low-latency voice translation pipeline that converts spoken input from one language to spoken output in another. This project aims for a sub-2-second end-to-end latency, making it suitable for near real-time applications.

## 🚀 Key Features

- **Speech-to-Text (STT)**: Powered by OpenAI's Whisper (Tiny model) for rapid transcription.
- **Machine Translation (MT)**: Utilizes Helsinki-NLP's Opus-MT models via Hugging Face Transformers for accurate linguistic conversion.
- **Text-to-Speech (TTS)**: Leverages Microsoft's `edge-tts` for natural-sounding, neural voice synthesis without heavy local compute requirements.
- **Multi-Device Support**: Automatically detects and uses CUDA, MPS (Metal), or CPU.
- **Low Latency**: Optimized to meet a < 2.0s total processing time for audio chunks.

## 🛠️ Tech Stack

- **Python 3.10+**
- **STT**: `openai-whisper`
- **MT**: `transformers`, `torch`
- **TTS**: `edge-tts`
- **Audio Processing**: `librosa`, `numpy`
- **Asyncio**: For asynchronous execution of TTS and overall pipeline coordination.

## 📁 Project Structure

```text
.
├── src/
│   └── pipeline.py       # Core pipeline implementation
├── data/                 # Input audio files for testing
├── outputs/              # Generated translated audio files
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Project configuration
```

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd demo
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

To run the smoke test and verify the pipeline:

```bash
python src/pipeline.py
```

This will:
1. Initialize the `VoiceTranslationPipeline`.
2. Load and cache the necessary models.
3. Process a sample audio file (`data/sample_text_input.wav`).
4. Perform EN -> ES translation.
5. Save the output to `outputs/smoke_test_output.wav`.
6. Output latency metrics to the console.

## 🌍 Supported Languages

The pipeline currently supports the following languages for both translation and synthesis:

| Code | Language | Neural Voice (Edge-TTS) |
| :--- | :--- | :--- |
| `en` | English | `en-US-GuyNeural` |
| `es` | Spanish | `es-ES-AlvaroNeural` |
| `fr` | French | `fr-FR-HenriNeural` |
| `de` | German | `de-DE-ConradNeural` |

## 📊 Performance Benchmarks

The pipeline is designed with a **2.0s latency target**. 
Current metrics tracked:
- **STT Time**: Whisper Tiny transcription latency.
- **MT Time**: Opus-MT translation latency.
- **TTS Time**: Edge-TTS synthesis and download time.
- **Total Latency**: Total end-to-end time.

---
*Created by [Antigravity](https://github.com/google-deepmind/antigravity)*
