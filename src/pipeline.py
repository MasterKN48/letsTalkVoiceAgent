import os
import gc
import time  # only if you plan to use it
from dotenv import load_dotenv

import torch
from huggingface_hub import snapshot_download

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# MLX-Audio imports
from mlx_audio.stt.utils import load_model as load_asr
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.tts.utils import load_model as load_tts
from mlx_audio.tts.generate import generate_audio


class VoiceTranslationPipeline:
    def __init__(self, use_gpu: bool = False):
        # Device selection (Torch only used for device/memory here)
        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # Project root and models directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        print("Initializing Native MLX-Audio Pipeline")
        print(f"Models directory: {self.models_dir}")

        load_dotenv()
        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

        # 1. STT: MLX-optimized Qwen3-ASR (or other ASR)
        self.asr_repo = os.getenv("ASR_MODEL", "mlx-community/Qwen3-ASR-0.6B-5bit")
        self.asr_path = self._ensure_local_model(self.asr_repo, "asr")
        self.asr_model = load_asr(self.asr_path)

        # 2. TTS: MLX-optimized Qwen3-TTS (or other TTS)
        self.tts_repo = os.getenv("TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit")
        self.tts_path = self._ensure_local_model(self.tts_repo, "tts")
        self.tts_model = load_tts(self.tts_path)

        # 3. MT: Local OpenAI-compatible API via LangChain
        api_base = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
        llm_model = os.getenv("LLM_MODEL", "qwen3.5-0.8b")
        print(f"Initializing Local Translation LLM at: {api_base} with model: {llm_model}")

        self.llm = ChatOpenAI(
            base_url=api_base,
            api_key="not-needed",  # local server usually ignores this
            model=llm_model,
            temperature=0,
        )

        self.translate_prompt = ChatPromptTemplate.from_template(
            "You are a professional translator. Translate the following text from "
            "{src_lang} to {tgt_lang}. Only return the translated text without any "
            "explanations.\n\n"
            "Text: {text}\n"
            "Translation:"
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _ensure_local_model(self, repo_id: str, sub_dir: str) -> str:
        """
        Downloads model from Hugging Face into a local subdirectory if not present,
        and returns that local path.
        """
        local_path = os.path.join(self.models_dir, sub_dir)
        if not os.path.exists(local_path) or not os.listdir(local_path):
            print(f"Downloading model {repo_id} to {local_path}...")
            snapshot_download(repo_id=repo_id, local_dir=local_path)
        else:
            print(f"Using local model: {local_path}")
        return local_path

    # -------------------------------------------------------------------------
    # Translation LLM
    # -------------------------------------------------------------------------
    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text from src_lang to tgt_lang using the local LLM.
        If languages are same or text is empty, return the original text.
        """
        if src_lang == tgt_lang or not text.strip():
            return text

        print(f"Translating via Local LLM: {src_lang} -> {tgt_lang}")
        chain = self.translate_prompt | self.llm
        response = chain.invoke({"src_lang": src_lang, "tgt_lang": tgt_lang, "text": text})
        return response.content.strip()

    # -------------------------------------------------------------------------
    # Text-to-Speech (TTS)
    # -------------------------------------------------------------------------
    async def text_to_speech(self, text: str, tgt_lang: str, output_path: str):
        """
        Generate speech audio from text using MLX-optimized Qwen3-TTS.
        output_path is treated as the directory where audio is written.
        """
        print(f"Generating TTS for language: {tgt_lang}")
        os.makedirs(output_path, exist_ok=True)

        # You can map tgt_lang -> lang_code if needed (e.g., "en" -> "a" for Kokoro).
        # For Qwen3-TTS, you often just leave it to auto-detect or use standard codes.
        generate_audio(
            model=self.tts_model,
            text=text,
            output_path=output_path,
            # Optional extras:
            # file_prefix="output",      # default prefix for filenames
            # lang_code="en",           # or map from tgt_lang
            # voice="...",              # if the model defines named voices
            # audio_format="wav",
            # join_audio=True,
            # verbose=False,
        )

    # -------------------------------------------------------------------------
    # Speech-to-Text (STT)
    # -------------------------------------------------------------------------
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio to text using MLX-optimized Qwen3-ASR.
        """
        print(f"Transcribing audio: {audio_path}")

        # Python API: generate_transcription(model=..., audio=...) style.[web:18][web:10]
        transcription = generate_transcription(
            model=self.asr_model,
            audio=audio_path,
            # You can also pass output_path to save .txt sidecar
            # output_path=None,
        )

        # Depending on version: can be an object with .text or a dict
        text = getattr(transcription, "text", None)
        if text is None and isinstance(transcription, dict):
            text = transcription.get("text", "")

        return (text or "").strip()

    # -------------------------------------------------------------------------
    # Memory cleanup
    # -------------------------------------------------------------------------
    def clear_memory(self):
        """
        Clears models from memory and triggers garbage collection.
        """
        print("Clearing models from memory...")

        if hasattr(self, "asr_model"):
            del self.asr_model

        if hasattr(self, "tts_model"):
            del self.tts_model

        if hasattr(self, "llm"):
            del self.llm

        gc.collect()

        if "mps" in self.device:
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass

        print("Memory cleanup complete.")
