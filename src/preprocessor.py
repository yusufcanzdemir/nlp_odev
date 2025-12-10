import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TurkishPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('turkish'))

    def turkish_lower(self, text):
        return text.replace('I', 'ı').replace('İ', 'i').lower() # I yı i diye dönüştürmemesi için

    def simple_stemmer(self, word):
        suffixes = ['lar', 'ler', 'nın', 'nin', 'daki', 'deki', 'dan', 'den', 'ı', 'i', 'u', 'ü']
        stem = word
        for suffix in suffixes: # Kelime bu ekle bitiyor mu?
            if stem.endswith(suffix) and len(stem) > len(suffix) + 2: # eki atınca kelime çok kısa oluyormu
                stem = stem[:-len(suffix)] # olmuyorsa eki at
                break
        return stem

    def clean_text(self, text, method="classic"):
        """
        method="classic": Stopwords atar, stemmer uygular (Naive Bayes için)
        method="deep": Sadece temizler, cümle yapısını bozmaz (BERT için)
        """
        text = re.sub(r'<.*?>', '', text) # HTML taglerini temizle (<br>, <div> vs.) - Regex ile
        text = self.turkish_lower(text)
        text = re.sub(r'[^\w\s]', '', text) # Sadece harf ve boşluk kalsın
        text = re.sub(r'\d+', '', text) # Sayıları sil
        
        tokens = word_tokenize(text)
        
        if method == "classic":
            tokens = [t for t in tokens if t not in self.stop_words] # Stopwords Removal (ve, ile, ama gibi kelimeleri at)
            tokens = [self.simple_stemmer(t) for t in tokens] # (Ekleri at)
            
        return " ".join(tokens)