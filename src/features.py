import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

def get_tfidf_features(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer # (x:şu anki verinin sayısal karşılığı, vectorizer: kural kitapçığı yeni veri gelirse karşılık ne üretilecek)

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

    # vector_size=100: Her kelimeyi 100 tane sayı ile ifade et. min_count = 1 bir tane geçen kelimeyi bile al.
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=5, min_count=1, workers=4)

    # bu yöntem keliemleri vektörleştirir ama biz cümle kullanıyoruz. cümledeki tüm kelime vektörlerinin ortalamasını alarak cümlenin vektörünü buluruz.
    # ortalama almak dışında max, toplam, ağırlıklı ort.(tf-idf), doc2Vec(paragraf id) gibi farklı yöntemlerde var.
    # ancak sınıflandırma için ortalama almak mantıklı ve yeterli.
    X = []
    for tokens in tokenized_corpus:
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vectors:
            sentence_vector = np.mean(vectors, axis=0)
        else:
            sentence_vector = np.zeros(vector_size)
        X.append(sentence_vector)
        
    return np.array(X), w2v_model