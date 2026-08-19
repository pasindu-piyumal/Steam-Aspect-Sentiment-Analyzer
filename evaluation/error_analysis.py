import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

DATASET = "data/processed/steam_reviews_clean.csv"

def analyze_errors():

    df = pd.read_csv(DATASET)

    train, test = train_test_split(df,test_size=0.2,random_state=42,stratify=df["sentiment"])

    vectorizer = TfidfVectorizer(max_features=10000,ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(train["clean_text"])

    X_test = vectorizer.transform(test["clean_text"])

    model = LinearSVC()
    model.fit(X_train,train["sentiment"])

    predictions = model.predict(X_test)

    test = test.copy()

    test["predicted"] = predictions

    errors = test[test["sentiment"] != test["predicted"]]

    print("Number of errors:",len(errors))
    print("\nExample errors:")
    print(errors[
            [
                "review_text",
                "sentiment",
                "predicted"
            ]
        ].head(20)
    )

    errors.to_csv(
        "results/error_analysis.csv",
        index=False
    )


if __name__ == "__main__":
    analyze_errors()