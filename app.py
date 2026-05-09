import streamlit as st
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Music Classifier", layout="centered")

# --- AI MODEL LOADING ---
@st.cache_resource
def load_multi_classifier():
    # This model is excellent for cross-language tasks
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_multi_classifier()

# --- UI HEADER ---
st.title("🎵 Global Music Classifier | مصنف الموسيقى")
st.write("Analyze songs in English or Arabic.")

# --- INPUT SECTION ---
# Clean, minimal input fields for the user
artist_name = st.text_input("Artist Name | اسم الفنان", placeholder="e.g., Amr Diab or Taylor Swift")
lyrics_text = st.text_area("Lyrics | كلمات الأغنية", placeholder="Enter lyrics here... / أدخل كلمات الأغنية هنا")

# Bilingual Labels for the AI to understand
genre_map = {
    "Pop / بوب": "pop music",
    "Rock / روك": "rock music",
    "Hip Hop / هيب هوب": "hip hop music",
    "Classic / كلاسيك": "classical music",
    "Shaabi / شعبي": "mahraganat or shaabi music",
    "Tarab / طرب": "arabic tarab music"
}

if st.button("Analyze / تحليل"):
    if artist_name and lyrics_text:
        combined_text = f"Artist: {artist_name}. Lyrics: {lyrics_text}"
        
        with st.spinner('Analyzing / جاري التحليل...'):
            # We send the English definitions to the model for better accuracy
            results = classifier(combined_text, candidate_labels=list(genre_map.values()))
        
        st.divider()
        st.subheader(":نتائج التحليل | Results") 

        # Map the English results back to our bilingual display labels
        reverse_map = {v: k for k, v in genre_map.items()}
        
        for i in range(3):
            eng_label = results['labels'][i]
            display_label = reverse_map[eng_label]
            score = results['scores'][i]
            
            st.write(f"**{display_label}** ({score*100:.1f}%)")
            st.progress(score)
            
    else:
        st.warning("Please fill in both fields | يرجى ملء الحقلين")

st.caption("Developed by Rawan Ayman Saber")