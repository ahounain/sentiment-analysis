from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
import librosa
import numpy as np
from huggingface_hub import login
import os
import shutil

# Define cache directory
CACHE_DIR = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/model_cache"

def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Created cache directory at {CACHE_DIR}")

def is_model_cached(model_dir):
    """Check if model is properly cached with all required files."""
    required_files = [
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
    ]
    return all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

def load_audio(file_path):
    """Load audio file."""
    audio, sr = librosa.load(file_path, sr=16000)
    return audio, sr
class Wav2VecTranscriber:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        ensure_cache_dir()
        
        # Set local path for model
        self.model_dir = os.path.join(CACHE_DIR, model_name.replace('/', '_'))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            if not is_model_cached(self.model_dir):
                print(f"Downloading and caching transcription model {model_name}...")
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
                
                print(f"Saving transcription model to {self.model_dir}")
                self.model.save_pretrained(self.model_dir)
                self.processor.save_pretrained(self.model_dir)
            else:
                print(f"Loading cached transcription model from {self.model_dir}")
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_dir, local_files_only=True)
                self.model = Wav2Vec2ForCTC.from_pretrained(self.model_dir, local_files_only=True)
        except Exception as e:
            print(f"Error loading transcription model: {e}")
            if os.path.exists(self.model_dir):
                print("Removing corrupted transcription cache and retrying...")
                shutil.rmtree(self.model_dir)
            
            print("Retrying transcription model download...")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            
            self.model.save_pretrained(self.model_dir)
            self.processor.save_pretrained(self.model_dir)
        
        self.model = self.model.to(self.device)
        
    def transcribe(self, audio_data):
        """Transcribe audio using wav2vec2."""
        try:
            inputs = self.processor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)

            return transcription[0]
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

def transcribe(audio_data):
    """Wrapper function for compatibility with main.py"""
    transcriber = Wav2VecTranscriber()
    lang_detector = LanguageDetector()
    
    language = lang_detector.detect_language(audio_data)
    transcription = transcriber.transcribe(audio_data)
    
    return transcription, language

def clear_cache():
    """Utility function to clear the cache."""
    if os.path.exists(CACHE_DIR):
        print(f"Clearing cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)