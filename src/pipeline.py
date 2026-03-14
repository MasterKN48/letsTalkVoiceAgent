import os
import torch
import whisper
import time
import gc
import edge_tts
from transformers import MarianMTModel, MarianTokenizer

class VoiceTranslationPipeline:
    def __init__(self, use_gpu=False):
        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # Calculate project root and models directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Whisper model directory
        self.whisper_cache = os.path.join(self.models_dir, "whisper")
        os.makedirs(self.whisper_cache, exist_ok=True)

        print(f"Initializing Pipeline on {self.device}")
        print(f"Models directory: {self.models_dir}")

        # 1. STT: Whisper Tiny (downloading to models/whisper)
        self.stt_model = whisper.load_model("tiny", device=self.device, download_root=self.whisper_cache)

        # 2. MT: Cached models and tokenizers
        self.mt_models = {}
        self.mt_tokenizers = {}
        self.mt_cache = os.path.join(self.models_dir, "mt")
        os.makedirs(self.mt_cache, exist_ok=True)

        # 3. TTS: Edge-TTS
        self.supported_langs = ["en", "es", "fr", "de"]

    def _get_mt_model(self, src, tgt):
        key = f"{src}-{tgt}"
        if key not in self.mt_models:
            model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            print(f"Loading MT model: {model_name}")
            
            # Using cache_dir to ensure models are saved in models/mt
            self.mt_tokenizers[key] = MarianTokenizer.from_pretrained(
                model_name, cache_dir=self.mt_cache
            )
            self.mt_models[key] = MarianMTModel.from_pretrained(
                model_name, cache_dir=self.mt_cache
            ).to(self.device)
            self.mt_models[key].eval()
        return self.mt_models[key], self.mt_tokenizers[key]

    def translate_text(self, text, src_lang, tgt_lang):
        if src_lang == tgt_lang or not text.strip():
            return text
        model, tokenizer = self._get_mt_model(src_lang, tgt_lang)
        tokens = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            translated = model.generate(**tokens, max_new_tokens=64, num_beams=1, do_sample=False)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    async def text_to_speech(self, text, tgt_lang, output_path):
        voices = {
            "en": "en-US-GuyNeural",
            "es": "es-ES-AlvaroNeural",
            "fr": "fr-FR-HenriNeural",
            "de": "de-DE-ConradNeural",
        }
        voice = voices.get(tgt_lang, "en-US-GuyNeural")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def load_audio_whisper(self, audio_path):
        import librosa
        import numpy as np

        # Load audio and resample to 16kHz as required by Whisper
        audio, _ = librosa.load(audio_path, sr=16000)
        return audio.astype(np.float32)

    def clear_memory(self):
        """Clears models from memory and triggers garbage collection."""
        print("Clearing models from memory...")
        
        # 1. Move models to CPU to break Metal acceleration links before deletion
        if hasattr(self, 'stt_model'):
            try:
                self.stt_model.to("cpu")
            except:
                pass
            del self.stt_model
            
        if hasattr(self, 'mt_models'):
            for key in list(self.mt_models.keys()):
                try:
                    self.mt_models[key].to("cpu")
                except:
                    pass
            self.mt_models.clear()
            del self.mt_models
            
        if hasattr(self, 'mt_tokenizers'):
            self.mt_tokenizers.clear()
            del self.mt_tokenizers

        # 2. Force Garbage Collection
        gc.collect()
        
        # 3. Clear GPU cache
        if "cuda" in self.device:
            torch.cuda.empty_cache()
        elif "mps" in self.device:
            # Explicitly clear MPS cache for Apple Silicon
            try:
                torch.mps.empty_cache()
            except AttributeError:
                # Older torch versions might not have this
                pass
            
        print("Memory cleanup complete.")
