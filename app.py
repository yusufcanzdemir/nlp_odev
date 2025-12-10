import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import numpy as np

from src.data_loader import get_dataset
from src.preprocessor import TurkishPreprocessor
from src.features import get_tfidf_features, get_bow_features, get_ngram_features, get_word2vec_features
from src.augmentation import DataAugmenter

st.set_page_config(page_title="NLP Studio", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    /* Sayfanın üst kısmına boşluk bırakıyoruz */
    .block-container {
        padding-top: 6rem; /* Burayı artırdım, artık kesik görünmeyecek */
        padding-bottom: 5rem;
    }
    /* Başlık stili */
    .app-header {
        font-size: 28px; 
        font-weight: 800; 
        color: #333; 
        margin-top: -20px; /* Başlığı kendi içinde biraz yukarı çektik */
    }
    /* Karanlık mod uyumu */
    @media (prefers-color-scheme: dark) {
        .app-header { color: #fff; }
    }
    /* Veri sayısı stili */
    .data-counter {
        text-align: right; 
        font-size: 18px; 
        font-weight: bold; 
        color: #FFFFFF;
        background-color: #28a745;
        padding: 8px 20px;
        border-radius: 20px;
        display: inline-block;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'df' not in st.session_state:
    st.session_state['df'] = get_dataset()
    st.session_state['clean_text'] = None
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=["Tarih/Saat", "Veri Sayısı", "Özellik", "Algoritma", "Parametre", "Doğruluk (%)"])
if 'model_registry' not in st.session_state:
    st.session_state['model_registry'] = {}

prep = TurkishPreprocessor()

col_head_1, col_head_2 = st.columns([8, 2])

with col_head_1:
    st.markdown('<div class="app-header">🧠 NLP Studio</div>', unsafe_allow_html=True)

with col_head_2:
    st.markdown(f"""
        <div style="text-align: right;">
            <span class="data-counter">🗃️ {len(st.session_state['df'])} Veri</span>
        </div>
    """, unsafe_allow_html=True)

tab_train, tab_aug, tab_predict = st.tabs(["⚙️ Model Eğitimi", "🧬 Veri İşlemleri", "🔮 Canlı Tahmin"])

with tab_train:
    col_settings, col_results = st.columns([1, 2], gap="medium")
    
    with col_settings:
        st.markdown("##### 🛠️ Ayarlar")
        
        vectorizer_type = st.selectbox("1. Vektörleştirme", ["TF-IDF", "BoW", "N-Gram", "Word2Vec"])
        
        n_val = 2
        if vectorizer_type == "N-Gram":
            n_val = st.slider("N Değeri:", 1, 4, 2)
        
        algo_type = st.selectbox("2. Algoritma", ["MaxEnt (Lojistik)", "Naive Bayes", "KNN"])
        
        k_val = 5
        if algo_type == "KNN":
            k_val = st.slider("K Komşu:", 1, 20, 5)
            
        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("Modeli Eğit 🚀", type="primary", use_container_width=True)

    with col_results:
        st.markdown("##### 📊 Sonuçlar")
        
        if train_btn:
            if vectorizer_type == "Word2Vec" and "Naive Bayes" in algo_type:
                st.error("❌ Word2Vec + Naive Bayes uyumsuzdur.")
            else:
                with st.spinner("Model eğitiliyor..."):
                    if st.session_state['clean_text'] is None:
                         st.session_state['clean_text'] = st.session_state['df']['text'].apply(lambda x: prep.clean_text(x, method="classic"))
                    
                    clean_text = st.session_state['clean_text']
                    y = st.session_state['df']['label']
                    
                    X = None
                    vec = None
                    try:
                        if vectorizer_type == "TF-IDF": X, vec = get_tfidf_features(clean_text)
                        elif vectorizer_type == "BoW": X, vec = get_bow_features(clean_text)
                        elif vectorizer_type == "N-Gram": X, vec = get_ngram_features(clean_text, n=n_val)
                        elif vectorizer_type == "Word2Vec": X, vec = get_word2vec_features(clean_text)
                    except Exception as e:
                        st.error(f"Hata: {e}")
                        st.stop()
                        
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    model = None
                    param_info = "-"
                    
                    if "MaxEnt" in algo_type: model = LogisticRegression(max_iter=1000)
                    elif "Naive Bayes" in algo_type: model = MultinomialNB()
                    elif "KNN" in algo_type: 
                        model = KNeighborsClassifier(n_neighbors=k_val)
                        param_info = f"k={k_val}"
                    
                    if vectorizer_type == "N-Gram": param_info = f"N={n_val}"

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    
                    st.success(f"✅ Başarı Oranı: **%{acc*100:.2f}**")
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    model_name = f"{algo_type} + {vectorizer_type} (%{acc*100:.1f}) - {timestamp}"
                    
                    st.session_state['model_registry'][model_name] = {
                        "model": model,
                        "vec": vec,
                        "type": vectorizer_type,
                        "acc": acc
                    }

                    with st.expander("🧩 Detaylı Analiz", expanded=True):
                        cm = confusion_matrix(y_test, y_pred)
                        labels = sorted(list(set(y)))
                        fig = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, colorscale='Blues', showscale=True)
                        st.plotly_chart(fig, use_container_width=True)

                    new_record = {
                        "Tarih/Saat": timestamp,
                        "Veri Sayısı": len(st.session_state['df']),
                        "Özellik": vectorizer_type,
                        "Algoritma": algo_type,
                        "Parametre": param_info,
                        "Doğruluk (%)": round(acc * 100, 2)
                    }
                    st.session_state['history'] = pd.concat([st.session_state['history'], pd.DataFrame([new_record])], ignore_index=True)

    if not st.session_state['history'].empty:
        st.markdown("---")
        st.markdown("##### 📜 Skor Tablosu")
        st.dataframe(
            st.session_state['history'].iloc[::-1].style.background_gradient(subset=['Doğruluk (%)'], cmap='Greens'), 
            use_container_width=True
        )

with tab_aug:
    col_aug_1, col_aug_2 = st.columns([1, 1])
    
    with col_aug_1:
        st.markdown("##### 🧬 Veri Çoğaltma")
        st.info("Eş anlamlı kelimeler kullanılarak veri seti yapay olarak büyütülür.")
        
        if st.button("Veriyi Çoğalt (+%20)", use_container_width=True):
            augmenter = DataAugmenter()
            with st.spinner("İşleniyor..."):
                old_len = len(st.session_state['df'])
                new_df = augmenter.augment_dataframe(st.session_state['df'])
                st.session_state['df'] = new_df
                st.session_state['clean_text'] = None
                diff = len(new_df) - old_len
            
            if diff > 0:
                st.success(f"Başarılı! +{diff} yeni veri.")
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Yeni veri üretilemedi.")

with tab_predict:
    if not st.session_state['model_registry']:
        st.warning("⚠️ Henüz hiç model eğitmediniz. Lütfen 'Model Eğitimi' sekmesine gidin.")
    else:
        st.markdown("##### 🧠 Tahmin Motoru")
        
        selected_model_name = st.selectbox(
            "Kullanılacak Modeli Seçin:",
            options=list(st.session_state['model_registry'].keys())
        )
        
        active_model_data = st.session_state['model_registry'][selected_model_name]
        
        col_input, col_pred = st.columns([2, 1])
        
        with col_input:
            text_input = st.text_area("Haber Metni:", height=120, placeholder="Metni buraya yapıştırın...")
            predict_btn = st.button("Analiz Et", type="primary", use_container_width=True)
            
        with col_pred:
            if predict_btn and text_input:
                clean_input = prep.clean_text(text_input, method="classic")
                
                vec = active_model_data['vec']
                model = active_model_data['model']
                feat_type = active_model_data['type']
                
                final_vector = None
                
                if feat_type == 'Word2Vec':
                    tokens = clean_input.split()
                    vectors = [vec.wv[word] for word in tokens if word in vec.wv]
                    final_vector = np.mean(vectors, axis=0).reshape(1, -1) if vectors else np.zeros((1, vec.vector_size))
                else:
                    final_vector = vec.transform([clean_input])
                
                prediction = model.predict(final_vector)[0]
                
                st.markdown("### Sonuç:")
                st.success(f"🏷️ **{prediction.upper()}**")
                st.caption(f"Kullanılan Model: {feat_type}")