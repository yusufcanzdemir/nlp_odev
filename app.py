import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.data_loader import get_dataset
from src.preprocessor import TurkishPreprocessor
from src.features import get_tfidf_features, get_bow_features, get_ngram_features, get_word2vec_features
from src.augmentation import DataAugmenter


st.set_page_config(page_title="NLP Modelleri", page_icon="🧠", layout="wide")
st.title("NLP Modelleri")
st.markdown("---")

if 'df' not in st.session_state:
    st.session_state['df'] = get_dataset()
    st.session_state['clean_text'] = None
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=["Tarih/Saat", "Veri Sayısı", "Özellik Çıkarımı", "Algoritma", "Doğruluk (%)"])
if 'best_model' not in st.session_state:
    st.session_state['best_model'] = None
    st.session_state['best_vectorizer'] = None
    st.session_state['best_feat_type'] = None

prep = TurkishPreprocessor()

st.info(f"**Aktif Veri Seti:** {len(st.session_state['df'])} Satır")

page = st.sidebar.radio("Menü:", ["Model Eğitimi & Kayıt", "Veri Çoğaltma", "Canlı Tahmin"])

if page == "Model Eğitimi & Kayıt":
    st.header("Model Konfigürasyonu")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Ayarlar")
        vectorizer_type = st.selectbox("Vektörleştirme Yöntemi", ["TF-IDF", "BoW", "N-Gram (Bigram)", "Word2Vec"])
        algo_type = st.selectbox("Algoritma Seçimi", ["MaxEnt (Lojistik Regresyon)", "Naive Bayes", "KNN"])
        
        train_btn = st.button("Modeli Eğit ve Kaydet", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Sonuç ve Analiz")
        
        if train_btn:
            # Word2Vec + Naive Bayes Kontrolü
            if vectorizer_type == "Word2Vec" and "Naive Bayes" in algo_type:
                st.error("HATA: Word2Vec ile Naive Bayes kullanılamaz (Negatif Değerler). Lütfen KNN seçin.")
            else:
                with st.spinner("Model eğitiliyor..."):
                    # 1. Temizlik (Cache mekanizması)
                    if st.session_state['clean_text'] is None:
                         st.session_state['clean_text'] = st.session_state['df']['text'].apply(lambda x: prep.clean_text(x, method="classic"))
                    
                    clean_text = st.session_state['clean_text']
                    y = st.session_state['df']['label']
                    
                    # 2. Vektörleştirme
                    X = None
                    vec = None
                    try:
                        if vectorizer_type == "TF-IDF":
                            X, vec = get_tfidf_features(clean_text)
                        elif vectorizer_type == "BoW":
                            X, vec = get_bow_features(clean_text)
                        elif vectorizer_type == "N-Gram (Bigram)":
                            X, vec = get_ngram_features(clean_text, n=2)
                        elif vectorizer_type == "Word2Vec":
                            X, vec = get_word2vec_features(clean_text)
                    except Exception as e:
                        st.error(f"Vektör Hatası: {e}")
                        st.stop()
                        
                    # 3. Eğitim
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    model = None
                    if "MaxEnt" in algo_type:
                        model = LogisticRegression(max_iter=1000)
                    elif "Naive Bayes" in algo_type:
                        model = MultinomialNB()
                    elif "KNN" in algo_type:
                        model = KNeighborsClassifier(n_neighbors=5)
                    
                    model.fit(X_train, y_train)
                    
                    # 4. Test
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    
                    st.success(f"Eğitim Tamamlandı! Başarı Oranı: **%{acc*100:.2f}**")
                    
                    # Tabloya Ekle
                    from datetime import datetime
                    new_record = {
                        "Tarih/Saat": datetime.now().strftime("%H:%M:%S"),
                        "Veri Sayısı": len(st.session_state['df']),
                        "Özellik Çıkarımı": vectorizer_type,
                        "Algoritma": algo_type,
                        "Doğruluk (%)": round(acc * 100, 2)
                    }
                    st.session_state['history'] = pd.concat([st.session_state['history'], pd.DataFrame([new_record])], ignore_index=True)
                    
                    # Bu modeli "Aktif Model" yap (Tahmin için)
                    st.session_state['best_model'] = model
                    st.session_state['best_vectorizer'] = vec
                    st.session_state['best_feat_type'] = vectorizer_type

    # --- GEÇMİŞ TABLOSU ---
    st.markdown("---")
    st.subheader("Eğitim Geçmişi (Leaderboard)")
    
    if not st.session_state['history'].empty:
        # Tabloyu göster (En son eklenen en üstte olsun diye ters çeviriyoruz)
        hist_df = st.session_state['history'].iloc[::-1]
        
        # Renkli Tablo (Doğruluk sütununu vurgula)
        st.dataframe(hist_df.style.background_gradient(subset=['Doğruluk (%)'], cmap='Greens'), use_container_width=True)
    else:
        st.info("Henüz bir eğitim yapmadınız. Yukarıdan ayarları seçip 'Eğit' butonuna basın.")

elif page == "🧬 Veri Çoğaltma":
    st.header("🧬 Veri Setini Genişlet")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mevcut Veri Sayısı", len(st.session_state['df']))
        
    with col2:
        if st.button("Veriyi Çoğalt (+ Eş Anlamlılar)", use_container_width=True):
            augmenter = DataAugmenter()
            with st.spinner("Veriler türetiliyor..."):
                new_df = augmenter.augment_dataframe(st.session_state['df'])
                st.session_state['df'] = new_df
                st.session_state['clean_text'] = None # Temizliği sıfırla
                
            st.success(f"İşlem Başarılı! Yeni Veri Sayısı: {len(new_df)}")
            st.rerun()

elif page == "Canlı Tahmin":
    st.header("Canlı Tahmin")
    
    if st.session_state['best_model'] is None:
        st.warning("Lütfen önce bir model eğitin!")
    else:
        st.info(f"Kullanılan Model: **{st.session_state['best_feat_type']}**")
        
        text_input = st.text_area("Haber Metni:", height=100)
        
        if st.button("Tahmin Et"):
            if text_input:
                import numpy as np
                clean_input = prep.clean_text(text_input, method="classic")
                
                vec = st.session_state['best_vectorizer']
                model = st.session_state['best_model']
                feat_type = st.session_state['best_feat_type']
                
                final_vector = None
                if feat_type == 'Word2Vec':
                    tokens = clean_input.split()
                    vectors = [vec.wv[word] for word in tokens if word in vec.wv]
                    if vectors:
                        final_vector = np.mean(vectors, axis=0).reshape(1, -1)
                    else:
                        final_vector = np.zeros((1, vec.vector_size))
                else:
                    final_vector = vec.transform([clean_input])
                
                prediction = model.predict(final_vector)[0]
                st.markdown(f"### Sonuç: **{prediction.upper()}**")