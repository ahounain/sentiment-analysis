from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import librosa
import numpy as np
import os
from evaluation.wav2vec import Wav2VecTranscriber

# Define cache directory
CACHE_DIR = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/model_cache"

class LanguageDetector:
    def __init__(self, model_name="facebook/mms-lid-2048"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = os.path.join(CACHE_DIR, model_name.replace('/', '_'))
        
        # Language mapping for MMS-LID model
        # These are the primary languages, there are more supported
        self.id2label = {
            'hin': 'Hindi',
            'eng': 'English',
            'ben': 'Bengali',
            'guj': 'Gujarati',
            'urd': 'Urdu',
            'tel': 'Telugu',
            'tam': 'Tamil',
            'mar': 'Marathi',
            'spa': 'Spanish',
            'fra': 'French',
            'deu': 'German',
            'ita': 'Italian',
            'jpn': 'Japanese',
            'kor': 'Korean',
            'cmn': 'Mandarin',
            'yue': 'Cantonese',
            'rus': 'Russian',
            'ara': 'Arabic',
            'tha': 'Thai',
            'vie': 'Vietnamese'
        }
        
        try:
            if not os.path.exists(self.model_dir):
                print(f"Downloading language detection model to {self.model_dir}...")
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=CACHE_DIR)
                self.model = AutoModelForAudioClassification.from_pretrained(model_name, cache_dir=CACHE_DIR)
                
                # Save locally
                self.model.save_pretrained(self.model_dir)
                self.feature_extractor.save_pretrained(self.model_dir)
            else:
                print("Loading cached language detection model...")
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_dir, local_files_only=True)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_dir, local_files_only=True)
            
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading language model: {e}")
            raise e
    
    def detect_language(self, audio_data):
        """
        Detect language from audio data.
        Returns tuple of (language_code, confidence, all_predictions)
        """
        try:
            # Prepare input features
            inputs = self.feature_extractor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(predictions, dim=-1).item()
            
            # Get confidence scores for all languages
            confidence_scores = predictions[0].cpu().numpy()
            
            # Get language labels from model config
            label_names = self.model.config.id2label
            
            # Create predictions list with readable names
            all_predictions = []
            for i, score in enumerate(confidence_scores):
                lang_code = label_names[i]
                lang_name = self.id2label.get(lang_code, lang_code)
                all_predictions.append((lang_name, float(score)))
            
            # Sort by confidence
            all_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get top prediction
            top_lang_code = label_names[predicted_id]
            top_confidence = confidence_scores[predicted_id]
            
            return top_lang_code, top_confidence, all_predictions
            
        except Exception as e:
            print(f"Error in language detection: {e}")
            return "eng", 0.0, [("English", 1.0)]
    
    def get_iso_code(self, language_name):
        """Convert language name to ISO code."""
        # Reverse mapping from language names to codes
        iso_codes = {v: k for k, v in self.id2label.items()}
        return iso_codes.get(language_name, "eng")

def transcribe(audio_data):
    """Wrapper function for compatibility."""
    transcriber = Wav2VecTranscriber()
    lang_detector = LanguageDetector()
    
    # Get language with confidence scores
    language_code, confidence, all_predictions = lang_detector.detect_language(audio_data)
    
    # Print detailed language analysis
    print("\nLanguage Detection Results:")
    print(f"Primary Language: {[pred[0] for pred in all_predictions if pred[1] == confidence][0]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nTop 5 Language Predictions:")
    for lang, conf in all_predictions[:5]:
        print(f"{lang}: {conf:.2%}")
    
    # Get transcription
    transcription = transcriber.transcribe(audio_data)
    
    return transcription, language_code