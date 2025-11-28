import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

def get_tfidf_features(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def get_bow_features(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def get_ngram_features(corpus, n=2):
    # n=2 ise Bigram (iki kelime), n=3 ise Trigram
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def get_word2vec_features(corpus, vector_size=100):
    tokenized_corpus = [sentence.split() for sentence in corpus]
    
    # 2. Word2Vec Modelini Eğit
    # vector_size=100: Her kelimeyi 100 tane sayı ile ifade et.
    # min_count=1: Sadece 1 kere geçen kelimeleri bile al (veri az olduğu için).
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=5, min_count=1, workers=4)
    
    # 3. Cümleleri Vektöre Çevir (Mean Embedding)
    # Cümledeki tüm kelime vektörlerinin ortalamasını alarak cümlenin vektörünü buluruz.
    X = []
    for tokens in tokenized_corpus:
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vectors:
            # Vektörlerin ortalamasını al (Ekseni koru)
            sentence_vector = np.mean(vectors, axis=0)
        else:
            # Eğer cümlede hiç bilinen kelime yoksa sıfır vektörü bas
            sentence_vector = np.zeros(vector_size)
        X.append(sentence_vector)
        
    return np.array(X), w2v_model