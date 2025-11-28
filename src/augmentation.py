import random
import pandas as pd

class DataAugmenter:
    def __init__(self):
        self.synonyms = {
            "maç": ["müsabaka", "karşılaşma"],
            "galibiyet": ["zafer", "kazanım"],
            "fenerbahçe": ["sarı kanaryalar", "fener", "sarı lacivertli ekip"],
            "galatasaray": ["aslan", "cimbom", "sarı kırmızılılar", "sarı kırmızılı ekip"],
            "beşiktaş": ["kara kartal","siyah beyazlılar","siyah beyazlı ekip"],
            "yenilgi": ["mağlubiyet"],
            "gol": ["sayı"],
            "rakip": ["hasım" , "ezeli rakip"],
            "final four": ["dörtlü final"],
            "antrenör": ["hoca", "teknik direktör", "teknik adam"],
            
            "yapay zeka": ["AI", "akıllı sistemler"],
            "bilgisayar": ["PC", "makine"],
            "internet": ["web", "ağ"],
            "yazılım": ["program", "uygulama"],
            "cihaz": ["aygıt", "donanım"],
            
            "para": ["nakit", "sermaye"],
            "düşüş": ["azalış", "gerileme"],
            "yükseliş": ["artış", "büyüme"],
            "şirket": ["firma", "kurum"],
            "fiyat": ["ücret", "bedel"],
            
            "kıyafet": ["giysi", "elbise"],
            "moda": ["trend", "akım"],
            "güzel": ["şık", "harika"],
            "yeni": ["güncel", "son model"],
            "önemli": ["kritik", "mühim"]
        }

    def augment_text(self, text):
        words = text.split()
        new_words = words.copy()
        changes_made = False
        
        for i, word in enumerate(words):
            # Noktalama temizliği yapmadan basit kontrol
            clean_word = word.lower().replace('.', '').replace(',', '')
            
            if clean_word in self.synonyms:
                # Rastgele bir eş anlamlı seç
                synonym = random.choice(self.synonyms[clean_word])
                new_words[i] = synonym # Kelimeyi değiştir
                changes_made = True
                
        if changes_made:
            return " ".join(new_words)
        else:
            return None # Değişiklik yapılamadıysa None dön

    def augment_dataframe(self, df):
        print(f"Orijinal Veri Sayısı: {len(df)}")
        new_rows = []
        
        for index, row in df.iterrows():
            original_text = row['text']
            label = row['label']
            
            # Veri türet
            aug_text = self.augment_text(original_text)
            
            if aug_text:
                new_rows.append({"text": aug_text, "label": label})
                
        if len(new_rows) > 0:
            new_df = pd.DataFrame(new_rows)
            # Orijinal ile yenileri birleştir
            combined_df = pd.concat([df, new_df], ignore_index=True)
            print(f"Türetilen Veri Sayısı: {len(new_rows)}")
            print(f"Toplam Yeni Veri Sayısı: {len(combined_df)}")
            return combined_df
        else:
            print("Hiç yeni veri türetilemedi (Eş anlamlı kelime bulunamadı).")
            return df