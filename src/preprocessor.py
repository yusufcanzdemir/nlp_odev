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
        return text.replace('I', 'ı').replace('İ', 'i').lower()

    def simple_stemmer(self, word):
        # Basit ek atıcı (Önceki yazdığımız mantık)
        suffixes = ['lar', 'ler', 'nın', 'nin', 'daki', 'deki', 'dan', 'den', 'ı', 'i', 'u', 'ü']
        stem = word
        for suffix in suffixes:
            if stem.endswith(suffix) and len(stem) > len(suffix) + 2:
                stem = stem[:-len(suffix)]
                break
        return stem

    def clean_text(self, text, method="classic"):
        """
        method="classic": Stopwords atar, stemmer uygular (Naive Bayes için)
        method="deep": Sadece temizler, cümle yapısını bozmaz (BERT için)
        """
        text = re.sub(r'<.*?>', '', text)
        text = self.turkish_lower(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        tokens = word_tokenize(text)
        
        if method == "classic":
            tokens = [t for t in tokens if t not in self.stop_words]
            tokens = [self.simple_stemmer(t) for t in tokens]
            
        return " ".join(tokens)