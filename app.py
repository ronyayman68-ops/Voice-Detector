import streamlit as st
from transformers import pipeline
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="Genre Sniper Elite", layout="centered")

# --- AI MODEL LOADING ---
@st.cache_resource
def load_models():
    # Model 1: Genre (BART)
    genre_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    # Model 2: Emotion/Mood (DistilRoBERTa)
    mood_classifier = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")
    return genre_classifier, mood_classifier

genre_ai, mood_ai = load_models()

# --- UI HEADER ---
st.title("Genre Sniper Elite")
st.write("Deep Style & Mood Analysis Pipeline")

# --- INPUT SECTION ---
col1, col2 = st.columns([1, 2])
with col1:
    artist = st.text_input("Artist Name", placeholder="e.g., Vivaldi")
with col2:
    lyrics = st.text_area("Lyrics / Style Description", placeholder="Enter text here...")

# Focused labels
labels = ["Classical Music", "Hip Hop / Rap Music"]

# --- EXECUTION ---
if st.button("Deep Scan"):
    if artist and lyrics:
        full_context = f"Artist: {artist}. Content: {lyrics}"
        
        with st.spinner('Analyzing patterns...'):
            # Prediction 1: Genre
            g_res = genre_ai(full_context, candidate_labels=labels, multi_label=False)
            # Prediction 2: Mood
            m_res = mood_ai(lyrics[:512]) # Take first 512 chars for speed
            
        st.divider()
        
        # UI Columns for Dual Results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader(" Style Result")
            top_genre = g_res['labels'][0]
            if "Classical" in top_genre:
                st.success(f"🎻 **CLASSICAL**\n\n{g_res['scores'][0]*100:.1f}% Match")
            else:
                st.info(f"🎤 **HIP HOP**\n\n{g_res['scores'][0]*100:.1f}% Match")

        with res_col2:
            st.subheader(" Mood Prediction")
            mood_label = m_res[0]['label'].upper()
            mood_score = m_res[0]['score']
            st.warning(f" **{mood_label}**\n\n{mood_score*100:.1f}% Match")
            
        # Comparison Bar
        st.write("---")
        st.write("Confidence Breakdown:")
        for label, score in zip(g_res['labels'], g_res['scores']):
            st.write(f"{label.split(' ')[0]}")
            st.progress(score)
            
    else:
        st.warning("Please provide both Artist and Lyrics.")

st.caption("Developed by Rawan Ayman Saber | Multi-Model NLP Pipeline")