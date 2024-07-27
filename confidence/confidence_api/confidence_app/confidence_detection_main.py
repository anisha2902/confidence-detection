import os
from confident_detection_helper1 import AudioProcessor

def main(audio_file_path, model_path):
    processor = AudioProcessor(model_path)
    prediction = processor.process_file(audio_file_path)
    if prediction is not None:
        print(f"File: {audio_file_path}, Prediction: {prediction}")
    else:
        print(f"Error processing file: {audio_file_path}")

# Example usage
if __name__ == "__main__":
    audio_file_path = 'Audio_Data/Non_Confident_2/nc.wav'
    model_path = 'svm_model.joblib'
    main(audio_file_path, model_path)
