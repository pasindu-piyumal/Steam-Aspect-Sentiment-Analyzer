import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,classification_report)

DATASET = "data/processed/steam_reviews_clean.csv"

def train_model():

    df = pd.read_csv(DATASET)

    X = df["clean_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    vectorizer = TfidfVectorizer(max_features=10000,ngram_range=(1, 2))

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)

    model.fit(X_train_tfidf,y_train)

    predictions = model.predict(X_test_tfidf)

    print("Accuracy:",accuracy_score(y_test,predictions))
    print(classification_report(y_test,predictions))

    os.makedirs("models_saved",exist_ok=True)
    joblib.dump(vectorizer,"models_saved/rf_tfidf.pkl")
    joblib.dump(model,"models_saved/random_forest.pkl")

if __name__ == "__main__":
    train_model()