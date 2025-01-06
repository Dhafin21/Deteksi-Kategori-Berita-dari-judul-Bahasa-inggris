import streamlit as st
import joblib
import re

# Memuat model dan TfidfVectorizer
model = joblib.load('/content/model_kategori_berita_pickle.pkl')
vectorizer = joblib.load('/content/vectorizer_kategori_berita_pickle.pkl')

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Fungsi untuk prediksi
def predict_category(input_text):
    cleaned_input = clean_text(input_text)
    input_tfidf = vectorizer.transform([cleaned_input])
    predicted_category = model.predict(input_tfidf)
    return predicted_category[0]

# Streamlit UI
st.title('Klasifikasi Kategori dari Judul Berita')
input_text = st.text_area("Masukan Judul Berita")

# Button to trigger sentiment analysis
if st.button('Analisis Judul Berita'):
    if input_text:
        predicted_category = predict_category(input_text)
        st.write(f"Kategori Berita adalah: {predicted_category}")
    else:
        st.write("Silakan masukkan Judul Berita.")
