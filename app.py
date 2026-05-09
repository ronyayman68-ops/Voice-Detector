import streamlit as st
from transformers import pipeline
import librosa
import torch

# --- PAGE CONFIG ---
st.set_page_config(page_title="VoiceAI Classifier", layout="centered")

# --- AI MODEL LOADING ---
@st.cache_resource
def load_audio_model():
    # Using a specialized model for Emotion Recognition in Speech
    return pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

classifier = load_audio_model()

# --- UI HEADER ---
st.title("🎙️ Voice Emotion Classifier")
st.write("Upload a voice recording to analyze the emotional tone.")

# --- FILE UPLOADER ---
audio_file = st.file_uploader("Upload Audio (wav, mp3):", type=["wav", "mp3"])

if audio_file is not None:
    # Display the audio player
    st.audio(audio_file, format='audio/wav')
    
    if st.button("Analyze Voice / تحليل الصوت"):
        with st.spinner('Processing audio signals...'):
            # Load audio file using librosa for better compatibility
            speech, sampling_rate = librosa.load(audio_file, sr=16000)
            
            # Run classification
            results = classifier(speech)
            
        st.divider()
        st.subheader(":نتائج تحليل الصوت") # Keeping your signature Arabic subheader
        
        # Display results with blue progress bars
        for result in results:
            label = result['label']
            score = result['score']
            
            st.write(f"**{label.capitalize()}**")
            st.progress(score)
            st.caption(f"Confidence: {score*100:.2f}%")
else:
    st.info("Please upload an audio file to start.")

st.caption("Developed by Rawan Ayman Saber | Data Engineer")