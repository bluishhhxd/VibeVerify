from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import librosa

# Load your trained model (e.g., SVM) and scaler
model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Process the uploaded audio file (extract features)
    features = extract_features(file)
    
    # Scale the features using the same scaler used during training
    scaled_features = scaler.transform([features])
    
    # Predict using the loaded model
    prediction = model.predict(scaled_features)
    
    # Return JSON response with predicted genre
    return jsonify({'genre': prediction[0]})

def extract_features(file):
    """ Extracts audio features from the uploaded audio file. """
    y, sr = librosa.load(file, sr=None)  # Load audio file with original sample rate
    
    # Extracting features similar to those in your dataset
    chroma_stft_mean = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    chroma_stft_var = librosa.feature.chroma_stft(y=y, sr=sr).var()
    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()
    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()
    spectral_bandwidth_mean = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_bandwidth_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()
    rolloff_mean = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y).var()

    harmony = librosa.effects.harmonic(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)

    perceptr = librosa.effects.percussive(y)
    perceptr_mean = np.mean(perceptr)
    perceptr_var = np.var(perceptr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_var = np.var(mfccs.T, axis=0)

    feature_vector = np.hstack([
        chroma_stft_mean, chroma_stft_var,
        rms_mean, rms_var,
        spectral_centroid_mean, spectral_centroid_var,
        spectral_bandwidth_mean, spectral_bandwidth_var,
        rolloff_mean, rolloff_var,
        zero_crossing_rate_mean, zero_crossing_rate_var,
        harmony_mean, harmony_var,
        perceptr_mean, perceptr_var,
        tempo,
        mfccs_mean, mfccs_var
    ])
    
    return feature_vector

if __name__ == '__main__':
    app.run(debug=True)