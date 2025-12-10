import pandas as pd
import os

# Sabitleri tanımlıyoruz:
# RAW_DATA_FOLDER: Ham metin dosyalarının olduğu yer (ttc3600 klasörü)
# CSV_FILE_PATH: İşlenmiş veriyi kaydedeceğimiz "hızlı erişim" dosyası
RAW_DATA_FOLDER = os.path.join('data', 'ttc3600')
CSV_FILE_PATH = os.path.join('data', 'dataset.csv')


def load_from_folders():
    print("TTC-3600 veri seti klasörlerden okunuyor.")

    if not os.path.exists(RAW_DATA_FOLDER):
        raise FileNotFoundError(
            f"HATA: '{RAW_DATA_FOLDER}' klasörü bulunamadı. Lütfen TTC-3600 klasörlerini data/ttc3600 içine atın.")

    data = []

    categories = os.listdir(RAW_DATA_FOLDER)  # sınıfları alır (ekonomi,siyaset,futbol...)

    for category in categories:
        category_path = os.path.join(RAW_DATA_FOLDER, category)

        if os.path.isdir(category_path):
            print(f" -> '{category}' kategorisi okunuyor...")

            for filename in os.listdir(category_path):  # tüm .txt leri gez
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_path, filename)

                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()

                        text = text.strip()

                        if text:
                            data.append({
                                "text": text,
                                "label": category
                            })

                    except Exception as e:
                        print(f"Hata (Dosya atlandı): {filename} -> {e}")

    df = pd.DataFrame(data)

    print(f"Toplam {len(df)} dosya okundu ve CSV olarak kaydedildi.")
    df.to_csv(CSV_FILE_PATH, index=False)

    return df


def get_dataset():
    if os.path.exists(CSV_FILE_PATH):
        print(f"Hazır veriseti yükleniyor: {CSV_FILE_PATH}")
        df = pd.read_csv(CSV_FILE_PATH)
        return df
    else:
        return load_from_folders()