# 🧠 Türkçe Metin Sınıflandırma ve NLP Projesi

Bu proje, Doğal Dil İşleme (NLP) teknikleri kullanılarak Türkçe haber metinlerini (Spor, Ekonomi, Teknoloji vb.) sınıflandıran yapay zeka uygulamasıdır. Verileri eş anlamlıları ile değiştirerek veri çoğaltabilir, aynı verinin farklı modellerle farkı doğruluk oranlarına sahip olduğunu görebilirsiniz.

## 🚀 Özellikler

* **Veri İşleme:** TTC-3600 veri seti entegrasyonu, temizlik, stopword removal.
* **Modeller:** Naive Bayes, KNN ve MaxEnt (Lojistik Regresyon).
* **Öznitelik Çıkarımı:** TF-IDF, BoW, N-Grams ve Word2Vec.

## 🛠️ Kurulum

1.  Projeyi klonlayın:
    ```bash
    git clone [https://github.com/yusufcanzdemir/Turkce-Metin-Siniflandirma-NLP.git](https://github.com/yusufcanzdemir/Turkce-Metin-Siniflandirma-NLP.git)
    cd Turkce-Metin-Siniflandirma-NLP
    ```

2.  Sanal ortamı kurun ve kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

3.  Uygulamayı çalıştırın:
    ```bash
    python main.py
    ```

## 📂 Veri Seti Hakkında
Proje TTC-3600 veri setini kullanmaktadır. Telif hakları ve boyut nedeniyle ham veri bu depoda yer almamaktadır.
Link: https://github.com/denopas/TTC-3600 
