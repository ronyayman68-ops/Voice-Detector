import streamlit as st
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Genre Sniper", layout="centered")

# --- AI MODEL LOADING (Optimized for Accuracy) ---
@st.cache_resource
def load_specialized_classifier():
    # Switching to a model that is better at emotional and stylistic nuances
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Using a robust fallback for genre specifically
@st.cache_resource
def load_genre_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_genre_model()

# --- UI HEADER ---
st.title("🎵 Precision Genre Detector")
st.write("Specialized in distinguishing **Classic** vs **Hip Hop** styles.")

# --- INPUT SECTION ---
artist = st.text_input("Artist Name", placeholder="e.g., Mozart or Kendrick Lamar")
lyrics = st.text_area("Lyrics / Description", placeholder="Enter the lines here...")

# Focused labels for higher accuracy
specific_genres = ["Classical Music", "Hip Hop / Rap Music"]

if st.button("Deep Scan Genre"):
    if artist and lyrics:
        # Combining data for context
        full_context = f"This is a song by {artist}. The style and lyrics are: {lyrics}"
        
        with st.spinner('Running deep analysis...'):
            # Multi-label set to False forces the AI to choose the most likely one
            res = classifier(full_context, candidate_labels=specific_genres, multi_label=False)
            
        st.divider()
        
        # Display the Winner
        top_genre = res['labels'][0]
        top_score = res['scores'][0]
        
        if "Classical" in top_genre:
            st.success(f"🎯 Prediction: **CLASSIC** ({top_score*100:.1f}%)")
        else:
            st.info(f"🔥 Prediction: **HIP HOP** ({top_score*100:.1f}%)")
            
        # Comparison Bar
        st.write("Style Comparison:")
        cols = st.columns(len(res['labels']))
        for i, label in enumerate(res['labels']):
            cols[i].write(label.split(" ")[0])
            cols[i].progress(res['scores'][i])
            
    else:
        st.warning("Please provide both artist and lyrics for an accurate scan.")

st.caption("Custom Model Logic by Rawan Ayman Saber")