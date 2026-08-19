import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)

DATASET = "data/processed/steam_reviews_clean.csv"

def evaluate():
    df = pd.read_csv(DATASET)

    X = df['clean_text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    models = {

        "Naive Bayes":
            MultinomialNB(),

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "SVM":
            LinearSVC(),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
    }

    results = []

    for name, model in models.items():
        print(f'training {name}')

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            'Presicion': precision_score(y_test, predictions, average="weighted"),
            'Recall': recall_score(y_test, predictions, average="weighted"),
            'F1': f1_score(y_test, predictions, average="weighted")
        })

    result_df = pd.DataFrame(results)

    print("\nModel Comparison")
    print(result_df)

    result_df.to_csv(
        "results/model_comparison.csv",
        index=False
    )


if __name__ == "__main__":
    evaluate()