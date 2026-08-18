import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():

    resources = ['stopwords', 'wordnet', 'omw-1.4']

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

STOP_WORDS = set(stopwords.words('english'))

LEMMATIZER = WordNetLemmatizer()

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text =re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [token for token in tokens if token not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    return " ".join(tokens)

if __name__ == "__main__":
    example = """
    The graphics are AMAZING!!! 😍
    But the servers keep crashing.
    """

    print("Original:")
    print(example)

    print("\nCleaned:")
    print(clean_text(example))