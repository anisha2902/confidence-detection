import os
import numpy as np
import librosa
import joblib

class AudioFeatureExtractor:
    def __init__(self, sr):
        self.sr = sr

    def extract_features(self, data):
        try:
            features = []

            # Feature 1: Speech rate (words per second)
            speech_rate = len(librosa.effects.split(data)) / librosa.get_duration(y=data)
            features.append(speech_rate)

            # Feature 2: Pitch variation (standard deviation of pitch)
            pitches, magnitudes = librosa.piptrack(y=data, sr=self.sr)
            pitch_std = np.std(pitches[pitches > 0])  # filtering out zero pitches if necessary
            features.append(pitch_std)

            # Feature 3: Energy (mean energy of the signal)
            energy = np.mean(librosa.feature.rms(y=data))
            features.append(energy)

            # Extract mean of the first 5 MFCCs
            mfccs = librosa.feature.mfcc(y=data, sr=self.sr, n_mfcc=12)
            mfccs_mean = np.mean(mfccs, axis=1)
            features.extend(mfccs_mean.tolist())

            # Feature 4: Zero Crossing Rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=data))
            features.append(zcr)

            # Feature 5: Spectral Flux
            spectral_flux = np.mean(librosa.onset.onset_strength(y=data, sr=self.sr))
            features.append(spectral_flux)

            # Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=self.sr)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            features.extend(spectral_contrast_mean.tolist())

            # Harmonic-to-Noise Ratio (HNR)
            harmonic, percussive = librosa.effects.hpss(y=data)
            hnr = np.mean(librosa.feature.rms(y=harmonic) / (librosa.feature.rms(y=percussive) + 1e-6))
            features.append(hnr)

            # Spectral Roll-off
            spectral_rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=data, sr=self.sr))
            features.append(spectral_rolloff_mean)

            return np.array(features).reshape(1, -1)
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            return None

class AudioProcessor:
    def __init__(self, model_path):
        model_tuple = joblib.load(model_path)
        self.scaler = model_tuple[1]
        self.model = model_tuple[0]

    def process_file(self, filepath):
        try:
            data, sr = librosa.load(filepath)
            feature_extractor = AudioFeatureExtractor(sr)
            features = feature_extractor.extract_features(data)
            if features is None:
                print("No features extracted")
                return None

            print("Extracted Features:", features)
            features = np.delete(features, [2, 12], axis=1)
            print("Features after deletion:", features)
            
            features_scaled = self.scaler.transform(features)
            print("Scaled Features:", features_scaled)
            
            prediction = self.model.predict(features_scaled)
            print("Prediction:", prediction)
            return prediction
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")
            return None
