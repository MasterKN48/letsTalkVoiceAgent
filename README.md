# Let's Talk: Real-time Voice Translation Agent

A high-performance, fully local, and open-source real-time voice translation agent built with **LangGraph** and **MLX-Audio**. This project specifically targets seamless **English to Hindi** (and vice versa) voice translation with a sub-2-second latency goal.

## 🚀 Key Features

- **Agentic Workflow**: Orchestrated by **LangGraph** for modular, stateful execution.
- **Native MLX Support**: Optimized for Apple Silicon using `mlx-audio` for ASR and TTS.
- **Local LLM Translation**: Uses high-performance local LLMs (like Qwen 3.5) for translation via OpenAI-compatible APIs.
- **Zero-Shot Voice Cloning**: Clones the source speaker's voice for the translated output using Qwen3-TTS.
- **Fully Local**: Runs entirely on your machine—no data leaves your device.
- **Resource Efficient**: Built-in memory management and cleanup nodes.

## 🛠️ Tech Stack

- **Orchestration**: `langgraph`
- **Speech-to-Text (STT)**: OpenAI Whisper Tiny (optimized via `openai-whisper`)
- **Machine Translation (MT)**: Local LLM (Qwen 3.5) via `langchain-openai`
- **Text-to-Speech (TTS)**: Qwen3-TTS (0.6B) via `mlx-audio`
- **Hardware Acceleration**: Apple Silicon (MPS) optimized

## 📁 Project Structure

```text
.
├── src/
│   ├── pipeline.py            # Core MLX & LLM logic
│   └── agent_langgraph.ipynb  # LangGraph implementation & demo
├── models/                    # Locally cached models
├── inputs/                    # Input audio files for testing
├── outputs/                   # Generated translated audio
├── agents.md                  # Detailed agent architecture documentation
├── README.md                  # Project overview
└── pyproject.toml             # Dependency management (uv)
```

## ⚙️ Quick Start

### 1. Prerequisites
- **Python 3.10+** (Recommended: 3.12)
- **uv**: Fast Python package manager
- **Local LLM Server**: LM Studio, Ollama, or vLLM running an OpenAI-compatible API.

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd letsTalkVoiceAgent

# Setup environment
uv sync
source .venv/bin/activate
```

### 3. Environment Variables
Copy `.env.example` to `.env` and configure your local LLM URL:
```bash
LOCAL_LLM_URL=http://localhost:1234/v1
LLM_MODEL=qwen3.5-0.8b
```

## 📖 Usage

Run the `agent_langgraph.ipynb` notebook to see the agent in action. It follows this flow:
1. **Transcribe**: Converts English voice to text.
2. **Translate**: Uses local LLM to translate text to Hindi.
3. **Synthesize**: Generates Hindi speech with your original voice cloned.

For details on the agent architecture, see [agents.md](./agents.md).

## 🗺️ Roadmap

- [ ] **FastAPI Integration**: Expose the agent via a REST API.
- [ ] **WebSocket Support**: Real-time streaming for continuous conversation.
- [ ] **Bi-directional Translation**: Full Hindi -> English support.
- [ ] **UI Frontend**: A clean web interface for interaction.

---
