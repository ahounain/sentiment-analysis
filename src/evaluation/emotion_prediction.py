import numpy as np
import librosa

# Define emotion mapping
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_audio_features(file_path):
  
    try:
        # Load and resample audio
        audio, sr = librosa.load(file_path, sr=22050)
        
        # Extract various features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        # Combine features
        features = np.vstack([mfccs, chroma, mel[:40, :], spectral_contrast])
        
        return features.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_default_metadata():
    """Create neutral metadata features for prediction."""
    # Create default one-hot encodings
    modality = np.array([1, 0, 0])  # default to full_av
    vocal_channel = np.array([1, 0])  # default to speech
    intensity = np.array([1, 0])  # default to normal
    statement = np.array([1, 0])  # default to first statement
    repetition = np.array([1, 0])  # default to first repetition
    gender = np.array([1, 0])  # default to male
    actor = np.zeros(24)  # default actor encoding
    actor[0] = 1  # set to first actor
    
    # Combine all metadata features
    return np.concatenate([
        modality, vocal_channel, intensity, statement, 
        repetition, gender, actor
    ])

def pad_sequences(audio_features, target_length=None):
    """Pad or truncate audio features to consistent length."""
    if target_length is None:
        target_length = 275  # Default length from training
        
    if len(audio_features) > target_length:
        return audio_features[:target_length]
    else:
        pad_width = ((0, target_length - len(audio_features)), (0, 0))
        return np.pad(audio_features, pad_width, mode='constant')

def predict_emotion(audio_file_path, model, label_encoder):
    """Predict emotion for a single audio file."""
    try:
        # Extract audio features
        audio_features = extract_audio_features(audio_file_path)
        if audio_features is None:
            raise Exception("Failed to extract audio features")
        
        # Pad sequences
        audio_features_padded = pad_sequences(audio_features)
        
        # Create batch dimension
        audio_features_batch = np.expand_dims(audio_features_padded, axis=0)
        
        # Get default metadata features
        metadata_features = create_default_metadata()
        metadata_features_batch = np.expand_dims(metadata_features, axis=0)
        
        # Make prediction
        prediction = model.predict(
            [audio_features_batch, metadata_features_batch],
            verbose=0
        )
        
        # Get prediction results
        emotion_index = np.argmax(prediction)
        emotion_code = label_encoder.inverse_transform([emotion_index])[0]
        emotion_word = EMOTION_LABELS.get(emotion_code, "Unknown")
        confidence = float(np.max(prediction))
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_emotions = []
        for idx in top_indices:
            emotion_code = label_encoder.inverse_transform([idx])[0]
            emotion_name = EMOTION_LABELS.get(emotion_code, "Unknown")
            confidence_score = float(prediction[0][idx])
            top_emotions.append((emotion_name, confidence_score))
        
        return emotion_word, confidence, top_emotions
        
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        return "Unknown", 0.0, []