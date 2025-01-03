import numpy as np
import librosa
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# Metadata mappings
MODALITY_MAP = {'01': 'full_av', '02': 'video', '03': 'audio'}
VOCAL_CHANNEL_MAP = {'01': 'speech', '02': 'song'}
EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
INTENSITY_MAP = {'01': 'normal', '02': 'strong'}
STATEMENT_MAP = {'01': 'kids', '02': 'dogs'}
REPETITION_MAP = {'01': 'first', '02': 'second'}
GENDER_MAP = {'odd': 'male', 'even': 'female'}

def extract_audio_features(file_path):
    """Extract multiple audio features."""
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

def parse_filename(filename):
    """Parse RAVDESS filename and extract all metadata."""
    parts = filename.replace('.wav', '').split('-')
    actor_num = int(parts[6])
    
    metadata = {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': parts[2],
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6],
        'gender': 'male' if actor_num % 2 != 0 else 'female'
    }
    
    return metadata

def create_metadata_features(metadata):
    """Convert metadata to numerical features."""
    # One-hot encode categorical variables
    modality = to_categorical(int(metadata['modality']) - 1, num_classes=3)
    vocal_channel = to_categorical(int(metadata['vocal_channel']) - 1, num_classes=2)
    intensity = to_categorical(int(metadata['intensity']) - 1, num_classes=2)
    statement = to_categorical(int(metadata['statement']) - 1, num_classes=2)
    repetition = to_categorical(int(metadata['repetition']) - 1, num_classes=2)
    gender = [1, 0] if metadata['gender'] == 'male' else [0, 1]
    actor = to_categorical(int(metadata['actor']) - 1, num_classes=24)
    
    # Combine all metadata features
    return np.concatenate([
        modality, vocal_channel, intensity, statement, 
        repetition, gender, actor
    ])

def load_dataset(dataset_path):
    """Load and prepare the RAVDESS dataset with both audio and metadata features."""
    audio_features = []
    metadata_features = []
    emotions = []
    
    print("Loading dataset...")
    for actor_dir in os.listdir(dataset_path):
        if not actor_dir.startswith('Actor_'):
            continue
            
        actor_path = os.path.join(dataset_path, actor_dir)
        for filename in os.listdir(actor_path):
            if not filename.endswith('.wav'):
                continue
                
            file_path = os.path.join(actor_path, filename)
            
            # Extract audio features
            audio_feat = extract_audio_features(file_path)
            if audio_feat is None:
                continue
                
            # Parse metadata
            metadata = parse_filename(filename)
            meta_feat = create_metadata_features(metadata)
            
            # Store features and label
            audio_features.append(audio_feat)
            metadata_features.append(meta_feat)
            emotions.append(metadata['emotion'])
    
    return audio_features, metadata_features, emotions

def pad_sequences(audio_features, max_length=None):
    """Pad or truncate audio features to consistent length."""
    if max_length is None:
        max_length = max(len(x) for x in audio_features)
        
    padded_features = []
    for feat in audio_features:
        if len(feat) > max_length:
            padded_features.append(feat[:max_length])
        else:
            pad_width = ((0, max_length - len(feat)), (0, 0))
            padded_features.append(np.pad(feat, pad_width, mode='constant'))
            
    return np.array(padded_features)

def build_model(audio_input_shape, metadata_input_shape, num_emotions):
    """Build a model that combines audio features and metadata."""
    # Audio input branch
    audio_input = Input(shape=audio_input_shape, name='audio_input')
    x1 = Conv1D(64, 3, activation='relu')(audio_input)
    x1 = MaxPooling1D(2)(x1)
    x1 = Conv1D(128, 3, activation='relu')(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = LSTM(64, return_sequences=True)(x1)
    x1 = LSTM(32)(x1)
    x1 = Dense(64, activation='relu')(x1)
    
    # Metadata input branch
    metadata_input = Input(shape=metadata_input_shape, name='metadata_input')
    x2 = Dense(32, activation='relu')(metadata_input)
    x2 = Dense(16, activation='relu')(x2)
    
    # Combine branches
    combined = Concatenate()([x1, x2])
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    
    # Output layer
    output = Dense(num_emotions, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=[audio_input, metadata_input], outputs=output)
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    return model

def train_model(dataset_path, model_save_path='emotion_model.keras', encoder_save_path='label_encoder.joblib'):
    """Main training function."""
    # Load dataset
    audio_features, metadata_features, emotions = load_dataset(dataset_path)
    
    # Prepare audio features
    padded_audio = pad_sequences(audio_features)
    metadata_features = np.array(metadata_features)
    
    # Prepare labels
    label_encoder = LabelEncoder()
    emotion_encoded = label_encoder.fit_transform(emotions)
    emotion_onehot = to_categorical(emotion_encoded)
    
    # Split data
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(padded_audio))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Create train/test sets
    X_audio_train = padded_audio[train_idx]
    X_meta_train = metadata_features[train_idx]
    y_train = emotion_onehot[train_idx]
    
    X_audio_test = padded_audio[test_idx]
    X_meta_test = metadata_features[test_idx]
    y_test = emotion_onehot[test_idx]
    
    # Build and train model
    model = build_model(
        audio_input_shape=(padded_audio.shape[1], padded_audio.shape[2]),
        metadata_input_shape=(metadata_features.shape[1],),
        num_emotions=len(np.unique(emotions))
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train
    history = model.fit(
        [X_audio_train, X_meta_train],
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(
        [X_audio_test, X_meta_test],
        y_test,
        verbose=0
    )
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save label encoder
    joblib.dump(label_encoder, encoder_save_path)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, label_encoder, history

if __name__ == "__main__":
    DATASET_PATH = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/src/training/RAVDESS"
    model, label_encoder, history = train_model(DATASET_PATH)