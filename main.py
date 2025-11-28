import sys
from src.data_loader import get_dataset
from src.preprocessor import TurkishPreprocessor
from src.features import get_tfidf_features, get_bow_features, get_ngram_features, get_word2vec_features
from src.model_trainer import train_naive_bayes, train_knn, train_maxent, predict_new_text
from src.augmentation import DataAugmenter

def main():
    print("=== NLP SINIFLANDIRMA SİSTEMİ ===")
    
    trained_model = None
    trained_vectorizer = None
    current_feature_type = 'tfidf'
    
    try:
        df = get_dataset()
        print(f"Veri yüklendi: {len(df)} satır.")
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return
    
    prep = TurkishPreprocessor()
    print("Metinler temizleniyor...")
    df['clean_text'] = df['text'].apply(lambda x: prep.clean_text(x, method="classic"))
    
    while True:
        print("\n=== ÖZELLİK ÇIKARIM MENÜSÜ ===")
        print("1. TF-IDF ile Eğit")
        print("2. BoW ile Eğit")
        print("3. N-GRAM ile Eğit")
        print("4. WORD2VEC ile Eğit")
        print("---------------------------")
        print("5. Canlı Tahmin Yap")
        print("6. Veri Setini Çoğalt (Augmentation)")
        print("7. Çıkış")
        
        secim = input("Seçiminiz: ")
        
        if secim in ['1', '2', '3', '4']:
            X = None
            try:
                if secim == '1':
                    X, trained_vectorizer = get_tfidf_features(df['clean_text'])
                    current_feature_type = 'tfidf'
                elif secim == '2':
                    X, trained_vectorizer = get_bow_features(df['clean_text'])
                    current_feature_type = 'bow'
                elif secim == '3':
                    print("N-Gram için N değerini girin (Varsayılan: 2)")
                    try:
                        n_val = int(input("N: "))
                    except:
                        n_val = 2
                    X, trained_vectorizer = get_ngram_features(df['clean_text'], n=n_val)
                    current_feature_type = 'ngram'
                elif secim == '4':
                    print("Word2Vec modeli eğitiliyor...")
                    X, trained_vectorizer = get_word2vec_features(df['clean_text'])
                    current_feature_type = 'word2vec'
            except Exception as e:
                print(f"\nHATA: Öznitelik çıkarılırken sorun oluştu: {e}")
                continue 

            if X is not None:
                print(f"Vektör Matrisi Hazır. Boyut: {X.shape}")
                if secim == '4':
                    print("BİLGİ: Word2Vec negatif değerler üretebildiği için otomatik olarak KNN seçildi.")
                    trained_model = train_knn(X, df['label'])
                else:
                    print("\n--- ALGORİTMA SEÇİN ---")
                    print("A: Naive Bayes")
                    print("B: KNN")
                    print("C: MaxEnt / Logistic Regression")
                    
                    algo = input("Seçim (A/B/C): ").upper()
                    
                    if algo == 'A':
                        trained_model = train_naive_bayes(X, df['label'])
                    elif algo == 'B':
                        trained_model = train_knn(X, df['label'])
                    elif algo == 'C':
                        trained_model = train_maxent(X, df['label'])
                    else:
                        print("Geçersiz seçim, varsayılan olarak MaxEnt kullanıldı.")
                        trained_model = train_maxent(X, df['label'])
                
                print("\n Model başarıyla eğitildi, şimdi (Canlı Tahmin) kullanabilirsiniz.")
            else:
                print("HATA: X matrisi oluşturulamadı.")

        elif secim == '5':
            if trained_model is None:
                print("UYARI: Önce 1-4 arası bir seçenekle model eğitmelisiniz!")
            else:
                predict_new_text(trained_model, trained_vectorizer, prep, current_feature_type)
                
        elif secim == '6':
            augmenter = DataAugmenter()
            df = augmenter.augment_dataframe(df)
            df['clean_text'] = df['text'].apply(lambda x: prep.clean_text(x, method="classic"))
            print(f"Güncel Veri Sayısı: {len(df)}")
            
        elif secim == '7':
            print("Çıkış yapıtınız.")
            sys.exit()

if __name__ == "__main__":
    main()