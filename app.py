import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import torch
import re
import string
from transformers import AutoTokenizer, AutoModel

# =========================================
# LOAD MODELS & DATA
# =========================================
@st.cache_resource
def load_models():
    svm_model = joblib.load('SVC_model.pkl')
    pca_model = joblib.load('pca_model.pkl')
    
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    bert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
    
    return svm_model, pca_model, tokenizer, bert_model

svm, pca, tokenizer, bert = load_models()

@st.cache_data
def load_data():
    df = pd.read_csv("clean_review.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# =========================================================
# PREPROCESSING & SENTIMENT PREDICTION FUNCTION
# =========================================================
def simple_cleaning(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prediksi_sentimen(teks):
    teks_bersih = simple_cleaning(teks)
    
    inputs = tokenizer(teks_bersih, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = bert(**inputs)
    cls_embedding = outputs.last_hidden_state[:,0,:].numpy()
    
    pca_embedding = pca.transform(cls_embedding)
    
    prediksi = svm.predict(pca_embedding)[0]
    
    label_map = {0: "Negatif(0)", 1: "Netral(1)", 2: "Positif(2)"}
    return label_map.get(prediksi, "Tidak diketahui")

# =========================================================
# WEBSITE MAIN DISPLAY
# =========================================================
st.title("Dashboard Analisis Sentimen Aplikasi")

st.subheader("Cek Sentimen Komentar Baru")
user_input = st.text_input("Ketik ulasan atau komentar di sini:")
if user_input:
    with st.spinner("Sedang menganalisis..."):
        hasil = prediksi_sentimen(user_input)
        if "Positif" in hasil:
            st.success(f"Sentimen: **{hasil}**")
        elif "Negatif" in hasil:
            st.error(f"Sentimen: **{hasil}**")
        else:
            st.info(f"Sentimen: **{hasil}**")

st.markdown("---")

# =========================================================
# DASHBOARD
# =========================================================
st.subheader("Filter Tren & Statistik Review")

min_date = df['date'].min()
max_date = df['date'].max()

if min_date == max_date:
    st.write(f"Data hanya tersedia untuk tanggal {min_date.date()}")
    filtered = df
else:
    start_date, end_date = st.date_input("Pilih Rentang Waktu:", [min_date, max_date])
    filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

col1, col2 = st.columns(2)
col1.metric("Total Reviews (Filtered)", len(filtered))
if not filtered.empty and 'rating' in filtered.columns:
    col2.metric("Rata-rata Rating", round(filtered['rating'].mean(), 2))

st.subheader("Sampel Data Review")
st.dataframe(filtered.head(20))

# --- KODE UNTUK DOWNLOAD OTOMATIS ---
from google.colab import files
files.download("app.py")
