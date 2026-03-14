# Real-time Voice Translation Pipeline

A high-performance, low-latency voice translation pipeline that converts spoken input from one language to spoken output in another. This project aims for a sub-2-second end-to-end latency, making it suitable for near real-time applications.

## 🚀 Key Features

- **STT**: Powered by OpenAI's Whisper (Tiny model).
- **Machine Translation (MT)**: Helsinki-NLP's Opus-MT models via Transformers.
- **Text-to-Speech (TTS)**: Microsoft's `edge-tts` for natural voice synthesis.
- **Agent Architecture**: Uses **LangGraph** for a stateful, modular workflow (STT -> MT -> TTS -> Cleanup).
- **Local Model Caching**: Models are stored in a local `models/` directory for persistent, low-latency access.
- **Resource Management**: Automatic GPU/RAM cleanup after processing via a `clear_memory` stage.

## 🛠️ Tech Stack

- **Python 3.10+**
- **Orchestration**: `langgraph`
- **STT**: `openai-whisper`
- **MT**: `transformers`, `torch`
- **TTS**: `edge-tts`

## 📁 Project Structure

```text
.
├── src/
│   ├── pipeline.py            # Core logic and model utilities
│   ├── agent_langgraph.ipynb  # LangGraph agent implementation
│   └── pipeline.ipynb         # Step-by-step documented notebook
├── models/                    # Locally cached models (Whisper, MarianMT)
├── data/                      # Input audio files for testing
├── outputs/                   # Generated translated audio files
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── pyproject.toml             # Project configuration
```

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd demo
   ```

2. **Set up the project with uv**:
   ```bash
   uv sync
   ```

3. **Activate the environment**:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   ```

## 📖 Usage

### Using the LangGraph Agent

Open the `agent_langgraph.ipynb` notebook to run the full pipeline as a stateful agent:

```bash
jupyter notebook src/agent_langgraph.ipynb
```

The agent orchestrates the following flow:

1. **STT Node**: Transcribes the audio.
2. **MT Node**: Translates the text.
3. **TTS Node**: Generates the audio output.
4. **Cleanup Node**: Frees up system memory (RAM/GPU).

## 🌍 Supported Languages

The pipeline currently supports the following languages for both translation and synthesis:

| Code | Language | Neural Voice (Edge-TTS) |
| :--- | :------- | :---------------------- |
| `en` | English  | `en-US-GuyNeural`       |
| `es` | Spanish  | `es-ES-AlvaroNeural`    |
| `fr` | French   | `fr-FR-HenriNeural`     |
| `de` | German   | `de-DE-ConradNeural`    |

## 📊 Performance Benchmarks

The pipeline is designed with a **2.0s latency target**.
Current metrics tracked:

- **STT Time**: Whisper Tiny transcription latency.
- **MT Time**: Opus-MT translation latency.
- **TTS Time**: Edge-TTS synthesis and download time.
- **Total Latency**: Total end-to-end time.

---
