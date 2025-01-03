import numpy as np
import librosa
import tensorflow.keras.models as models
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import necessary functions
from evaluation.wav2vec import Wav2VecTranscriber
from evaluation.languagedetection import LanguageDetector
from evaluation.emotion_prediction import predict_emotion, EMOTION_LABELS
from evaluation.bertanalysis import analyze_sentiment

def main():
    # Initialize models
    transcriber = Wav2VecTranscriber()
    lang_detector = LanguageDetector()
    
    # Paths to emotion model and label encoder
    MODEL_PATH = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/src/training/emotion_model.keras"
    ENCODER_PATH = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/src/training/label_encoder.joblib"

    # Specify the path to your audio file
    audio_file_path = "audio.wav"

    # Load the trained emotion model and label encoder
    try:
        model = models.load_model(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        print(f"Error loading model or label encoder: {e}")
        return

    # Process audio with Wav2Vec and language detection
    audio_data, sr = librosa.load(audio_file_path, sr=16000)
    
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
    print(f"\nTranscription: {transcription}")



    # Predict emotion from the audio
    try:
        emotion, confidence, top_emotions = predict_emotion(audio_file_path, model, label_encoder)
        print("\nEmotion Analysis Results:")
        print(f"Primary Emotion: {emotion} (Confidence: {confidence:.2%})")
        print("\nTop 3 Emotions:")
        for emotion, conf in top_emotions:
            print(f"{emotion}: {conf:.2%}")
    except Exception as e:
        print(f"Error predicting emotion: {e}")

if __name__ == "__main__":
    main()