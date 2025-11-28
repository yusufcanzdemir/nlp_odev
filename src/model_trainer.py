from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_naive_bayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
    # stratify: veriseti sınıf dağılımını korur
    
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Naive Bayes Sonuçları ---")
    print(f"Doğruluk (Accuracy): %{acc*100:.2f}")
    print(classification_report(y_test, y_pred))
    
    return model

def train_knn(X, y, k=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- KNN (k={k}) Sonuçları ---")
    print(f"Doğruluk (Accuracy): %{acc*100:.2f}")
    
    return model

def train_maxent(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- MaxEnt (Logistic Regression) Sonuçları ---")
    print(f"Doğruluk (Accuracy): %{acc*100:.2f}")
    
    print(classification_report(y_test, y_pred))
    
    return model


def predict_new_text(model, vectorizer, prep_class, feature_type='tfidf'):
    print(f"\n--- TAHMİN MODU ({feature_type}) ---")
    print("Çıkmak için 'q' yazın.")
    
    while True:
        text = input("\nBir cümle girin: ")
        if text.lower() == 'q':
            break
            
        clean_text = prep_class.clean_text(text, method="classic")
        
        # Vektörleştirme (Model tipine göre değişir)
        if feature_type == 'word2vec':
            # Word2Vec işlemi manuel yapılır
            tokens = clean_text.split()
            # vectorizer burada aslında w2v_model'dir
            vectors = [vectorizer.wv[word] for word in tokens if word in vectorizer.wv]
            if vectors:
                vectorized_text = np.mean(vectors, axis=0).reshape(1, -1)
            else:
                vectorized_text = np.zeros((1, vectorizer.vector_size))
        else:
            # Klasik Scikit-Learn (TF-IDF, BoW, N-Gram)
            vectorized_text = vectorizer.transform([clean_text])
        
        prediction = model.predict(vectorized_text)
        print(f"-> Tahmin: {prediction[0]}")