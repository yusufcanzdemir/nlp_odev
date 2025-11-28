import pandas as pd
import os

RAW_DATA_FOLDER = os.path.join('data', 'ttc3600')
CSV_FILE_PATH = os.path.join('data', 'dataset.csv')

def load_from_folders():
    print("TTC-3600 veri seti klasörlerden okunuyor.")
    
    if not os.path.exists(RAW_DATA_FOLDER):
        raise FileNotFoundError(f"HATA: '{RAW_DATA_FOLDER}' klasörü bulunamadı. Lütfen TTC-3600 klasörlerini data/ttc3600 içine atın.")

    data = []
    
    categories = os.listdir(RAW_DATA_FOLDER)
    
    for category in categories:
        category_path = os.path.join(RAW_DATA_FOLDER, category)
        
        if os.path.isdir(category_path):
            print(f" -> '{category}' kategorisi okunuyor...")
            
            for filename in os.listdir(category_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_path, filename)
                    
                    try:
                        # Encoding sorunu olmaması için 'utf-8' veya 'windows-1254' deneriz
                        # TTC-3600 genelde 'windows-1254' (ANSI) olabilir, ama önce utf-8 deneriz.
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                            
                        text = text.strip()
                        
                        if text:
                            data.append({
                                "text": text,
                                "label": category  # Klasör adı etikettir (spor, ekonomi vb.)
                            })
                            
                    except Exception as e:
                        print(f"Hata (Dosya atlandı): {filename} -> {e}")

    df = pd.DataFrame(data)
    
    print(f"Toplam {len(df)} dosya okundu ve CSV olarak kaydedildi.")
    df.to_csv(CSV_FILE_PATH, index=False)
    
    return df

def get_dataset():
    if os.path.exists(CSV_FILE_PATH):
        print(f"Veri seti yükleniyor (Cache): {CSV_FILE_PATH}")
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            if len(df) < 10: 
                print("CSV dosyası çok küçük veya boş, yeniden taranıyor...")
                return load_from_folders()
            return df
        except:
            return load_from_folders()
            
    else:
        return load_from_folders()

"""def get_dataset(file_path='data/dataset.csv'):
    if not os.path.exists(file_path):
        # Belki kod src içinden çağrılıyordur, bir üst klasöre bak
        file_path = os.path.join('..', 'data', 'dataset.csv')
    
    if os.path.exists(file_path):
        print(f"Veri seti yükleniyor: {file_path}")
        df = pd.read_csv(file_path)
        df = df.dropna()
        return df
    else:
        raise FileNotFoundError(f"HATA: {file_path} bulunamadı.")"""