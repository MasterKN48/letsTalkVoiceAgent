import os
import gc
import time
from dotenv import load_dotenv

import torch
from huggingface_hub import snapshot_download

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Audio utils
import librosa
import soundfile as sf  # librosa depends on this; used for writing clips.[web:63]

# MLX-Audio
from mlx_audio.stt.utils import load_model as load_asr
from mlx_audio.stt.generate import generate_transcription
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
        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

        # 1. STT model (e.g., Qwen3-ASR)
        self.asr_repo = os.getenv("ASR_MODEL", "mlx-community/Qwen3-ASR-0.6B-5bit")
        self.asr_path = self._ensure_local_model(self.asr_repo, "asr")
        self.asr_model = load_asr(self.asr_path)

        # 2. TTS model (Qwen3-TTS 3-second voice clone capable).[web:45][web:52][web:55]
        self.tts_repo = os.getenv("TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit")
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
    # NEW: voice clone preparation
    # -------------------------------------------------------------------------
    def prepare_voice_clone(self, src_audio_path: str, work_dir: str) -> str | None:
        """
        If the source audio is > 3 seconds, create a 3-second clip for voice cloning.
        Returns path to the clipped reference audio, or None to use default voice.
        """
        # Fast duration check straight from file.[web:57][web:59]
        try:
            duration = librosa.get_duration(path=src_audio_path)
        except TypeError:
            # Fallback for older librosa versions that use `filename=`
            duration = librosa.get_duration(filename=src_audio_path)

        print(f"Source audio duration: {duration:.2f}s")

        if duration <= 3.0:
            print("Duration <= 3s; using default TTS voice (no clone).")
            return None

        print("Duration > 3s; creating 3-second reference clip for voice cloning.")
        # Load only the first 3 seconds at native sample rate.[web:60][web:66]
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
        response = chain.invoke({"src_lang": "src_lang", "tgt_lang": tgt_lang, "text": text})
        return response.content.strip()

    # -------------------------------------------------------------------------
    # TTS with optional voice cloning
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
        }
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:  # <— key change
                kwargs["ref_text"] = ref_text

        generate_audio(**kwargs)

    # -------------------------------------------------------------------------
    # STT
    # -------------------------------------------------------------------------
    def transcribe_audio(self, audio_path: str) -> str:
        print(f"Transcribing audio: {audio_path}")
        transcription = generate_transcription(
            model=self.asr_model,
            audio=audio_path,
        )
        text = getattr(transcription, "text", None)
        if text is None and isinstance(transcription, dict):
            text = transcription.get("text", "")
        return (text or "").strip()

    # -------------------------------------------------------------------------
    # Memory cleanup
    # -------------------------------------------------------------------------
    def clear_memory(self):
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
