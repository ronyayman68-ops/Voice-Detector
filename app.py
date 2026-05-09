import streamlit as st
from transformers import pipeline
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="Genre Sniper Elite", layout="centered")

# --- AI MODEL LOADING ---
@st.cache_resource
def load_models():
    # BART-Large for high-accuracy genre classification
    genre_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return genre_classifier

classifier = load_models()

# --- UI HEADER ---
st.title("🎯 Genre Sniper Elite")
st.write("Professional-grade distinction between **Classical** and **Hip Hop**.")

# --- INPUT SECTION ---
col1, col2 = st.columns([1, 2])
with col1:
    artist = st.text_input("Artist Name", placeholder="e.g., Bach")
with col2:
    lyrics = st.text_area("Lyrics / Style Description", placeholder="Enter text here...")

# High-precision labels
labels = ["Classical Music", "Hip Hop / Rap Music"]

# --- HELPER FUNCTION: Explainability ---
def get_keywords(text, genre):
    # Data Engineering logic: manual keyword check to explain AI decision
    hip_hop_hits = ["beat", "rhyme", "flow", "street", "mic", "rap", "hustle", "city"]
    classic_hits = ["orchestra", "symphony", "piano", "sonata", "violin", "composer", "opera"]
    
    found = []
    check_list = hip_hop_hits if "Hip Hop" in genre else classic_hits
    for word in check_list:
        if re.search(r'\b' + word + r'\b', text.lower()):
            found.append(word)
    return found

# --- EXECUTION ---
if st.button("Deep Scan Genre"):
    if artist and lyrics:
        full_context = f"Artist: {artist}. Content: {lyrics}"
        
        with st.spinner('Analyzing stylistic patterns...'):
            res = classifier(full_context, candidate_labels=labels, multi_label=False)
            
        st.divider()
        
        # Results Logic
        top_genre = res['labels'][0]
        top_score = res['scores'][0]
        
        # UI Feedback
        if "Classical" in top_genre:
            st.success(f"🎻 Prediction: **CLASSICAL** ({top_score*100:.1f}%)")
        else:
            st.info(f"🎤 Prediction: **HIP HOP** ({top_score*100:.1f}%)")

        # Explainability Section
        keywords = get_keywords(lyrics, top_genre)
        if keywords:
            st.write(f"**Detected Style Markers:** {', '.join(keywords)}")
            
        # Probability Breakdown
        st.write("---")
        st.write("Confidence Breakdown:")
        for label, score in zip(res['labels'], res['scores']):
            st.write(f"{label.split(' ')[0]}")
            st.progress(score)
            
    else:
        st.warning("Please provide both Artist and Lyrics.")

st.caption("Developed by Rawan Ayman Saber | Precision NLP Pipeline")