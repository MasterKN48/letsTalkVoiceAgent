import os
import gc
import time
import asyncio
from dotenv import load_dotenv
import torch

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import snapshot_download

# Audio utils
import librosa
import soundfile as sf  # librosa depends on this; used for writing clips.

# mlx-whisper for ASR (replace mlx_audio STT)
os.environ["HF_HOME"] = os.getenv("HF_HOME", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
import whisper

# MLX-Audio
from mlx_audio.tts.utils import load_model as load_tts
from mlx_audio.tts.generate import generate_audio


class VoiceTranslationPipeline:
    def __init__(self, use_gpu: bool = False):
        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        print("Initializing Native MLX-Audio Pipeline")
        print(f"Models directory: {self.models_dir}")

        load_dotenv()

        # 1. STT model
        ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-tiny")
        self.asr_path = self._ensure_local_model(ASR_MODEL, "asr")
        # Load Whisper model from your models/asr directory
        print(f"Loading Whisper tiny from: {self.asr_path}")
        self.whisper_model = whisper.load_model(
            name="tiny",
            download_root=self.asr_path,  # saves/loads from models/asr
            device=self.device,
        )

        # 2. TTS model (Qwen3-TTS 3-second voice clone capable).
        self.tts_repo = os.getenv(
            "TTS_MODEL",
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        )
        self.tts_path = self._ensure_local_model(self.tts_repo, "tts")
        self.tts_model = load_tts(self.tts_path)

        # 3. Local translation LLM
        api_base = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
        llm_model = os.getenv("LLM_MODEL", "qwen3.5-0.8b")
        print(f"Initializing Local Translation LLM at: {api_base} with model: {llm_model}")

        self.llm = ChatOpenAI(
            base_url=api_base,
            api_key="not-needed",
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
        local_path = os.path.join(self.models_dir, sub_dir)
        if not os.path.exists(local_path) or not os.listdir(local_path):
            print(f"Downloading model {repo_id} to {local_path}...")
            snapshot_download(repo_id=repo_id, local_dir=local_path)
        else:
            print(f"Using local model: {local_path}")
        return local_path

    # -------------------------------------------------------------------------
    # Voice clone preparation
    # -------------------------------------------------------------------------
    def prepare_voice_clone(self, src_audio_path: str, work_dir: str) -> str | None:
        """
        If the source audio is > 3 seconds, create a 3-second clip for voice cloning.
        Returns path to the clipped reference audio, or None to use default voice.
        """
        # Fast duration check straight from file.
        try:
            duration = librosa.get_duration(path=src_audio_path)
        except TypeError:
            duration = librosa.get_duration(filename=src_audio_path)

        print(f"Source audio duration: {duration:.2f}s")

        if duration <= 3.0:
            print("Duration <= 3s; using default TTS voice (no clone).")
            return None

        print("Duration > 3s; creating 3-second reference clip for voice cloning.")
        # Load only the first 3 seconds at native sample rate.
        y, sr = librosa.load(src_audio_path, sr=None, offset=0.0, duration=3.0)

        os.makedirs(work_dir, exist_ok=True)
        ref_path = os.path.join(work_dir, "voice_ref.wav")
        sf.write(ref_path, y, sr)
        print(f"Wrote 3s reference clip to: {ref_path}")
        return ref_path

    # -------------------------------------------------------------------------
    # Translation LLM
    # -------------------------------------------------------------------------
    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if src_lang == tgt_lang or not text.strip():
            return text

        print(f"Translating via Local LLM: {src_lang} -> {tgt_lang}")
        chain = self.translate_prompt | self.llm
        # BUG FIX: pass src_lang variable, not literal "src_lang"
        response = chain.invoke({"src_lang": src_lang, "tgt_lang": tgt_lang, "text": text})
        return response.content.strip()

    # -------------------------------------------------------------------------
    # TTS with optional voice cloning (offloaded to thread)
    # -------------------------------------------------------------------------
    async def text_to_speech(
        self,
        text: str,
        tgt_lang: str,
        output_path: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ):
        """
        Generate speech audio from text.
        If ref_audio is provided and long enough, Qwen3-TTS will clone that voice.
        If ref_text is also provided, MLX-Audio will NOT try to transcribe ref_audio itself.
        `output_path` is treated as an output directory.
        """
        print(
            f"Generating TTS for language: {tgt_lang} "
            f"(voice clone: {'ON' if ref_audio else 'OFF'})"
        )
        os.makedirs(output_path, exist_ok=True)

        kwargs = {
            "model": self.tts_model,
            "text": text,
            "output_path": output_path,
            "audio_format": "wav",
            # You can add stream=True, streaming_interval, etc. if you want streaming.
        }
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text

        # Run heavy TTS in a worker thread so we don't block the event loop.
        def _run_tts():
            generate_audio(**kwargs)

        await asyncio.to_thread(_run_tts)

    # -------------------------------------------------------------------------
    # STT
    # -------------------------------------------------------------------------
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using OpenAI Whisper tiny from models/asr.
        Always outputs English text.
        """
        print(f"Transcribing with OpenAI Whisper: {audio_path}")

        # Use MPS acceleration if available, CPU fallback
        fp16 = torch.backends.mps.is_available()

        result = self.whisper_model.transcribe(
            audio_path,
            language="en",  # Force English transcription
            fp16=fp16,  # MPS acceleration if available
            verbose=False,
            word_timestamps=False,
        )

        return result["text"].strip()

    # -------------------------------------------------------------------------
    # Memory cleanup
    # -------------------------------------------------------------------------
    def clear_memory(self):
        print("Clearing models from memory...")

        if hasattr(self, "whisper_model"):
            del self.whisper_model
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
