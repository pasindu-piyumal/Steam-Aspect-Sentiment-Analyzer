import pandas as pd
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from preprocessing.clean_text import clean_text


INPUT_FILE = "data/raw/steam_reviews_raw.csv"

OUTPUT_FILE = "data/processed/steam_reviews_clean.csv"


def prepare_dataset():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original records:", len(df))

    df = df.dropna(subset=["review_text"])

    df = df.drop_duplicates(subset=["review_text"])

    print("Cleaning reviews...")
    df["clean_text"] = df["review_text"].apply(clean_text)

    df = df[df["clean_text"].str.len() > 0]

    df["sentiment"] = df["recommended"].map({
        True: "positive",
        False: "negative"
    })

    os.makedirs("data/processed",exist_ok=True)

    df.to_csv(OUTPUT_FILE,index=False)

    print(f"Processed dataset saved to {OUTPUT_FILE}")

    print("\nDataset:")
    print(df.head())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

if __name__ == "__main__":
    prepare_dataset()