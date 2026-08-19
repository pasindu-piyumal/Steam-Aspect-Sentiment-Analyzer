import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

DATASET = "data/processed/steam_reviews_clean.csv"


def confusion_matrix_create():

    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(DATASET)

    df = df[["clean_text", "sentiment"]].dropna()

    df["sentiment"] = df["sentiment"].str.lower().str.strip()

    df["sentiment"] = df["sentiment"].replace({
        "posistive": "positive"
    })

    print("Sentiment classes:")
    print(df["sentiment"].value_counts())
    print()

    df = df[df["sentiment"].isin([
        "negative",
        "neutral",
        "positive"
    ])]


    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    labels = [
        "negative",
        "neutral",
        "positive"
    ]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("Confusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot()

    plt.title("SVM Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    confusion_matrix_create()