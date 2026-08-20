import re
from textblob import TextBlob

from aspect_analysis.aspect_extraction import extract_aspects, ASPECTS


def get_aspect_keywords(aspect):

    keywords = ASPECTS.get(aspect.lower(), [aspect])
    return sorted(keywords, key=len, reverse=True)


def get_sentiment(text):

    text = text.strip()

    if not text:
        return {
            "sentiment": "neutral",
            "polarity": 0.0
        }

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        sentiment = "positive"

    elif polarity < -0.1:
        sentiment = "negative"

    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "polarity": round(polarity, 3)
    }


def split_sentences(text):
    """
    Split review into individual sentences.
    """

    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def aspect_exists(sentence, aspect):

    keywords = get_aspect_keywords(aspect)

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
        re.IGNORECASE
    )

    return pattern.search(sentence)


def extract_aspect_context(sentence, aspect):

    match = aspect_exists(sentence, aspect)
    if not match:
        return sentence

    chunks = re.split(
        r",|;|\s(?:and|but|yet|however|though|while)\s",
        sentence,
        flags=re.IGNORECASE
    )

    for chunk in chunks:
        if aspect_exists(chunk, aspect):
            return chunk.strip()

    return sentence


def analyze_aspects(text):
    """Analyze sentiment for each detected aspect safely."""
    aspects = extract_aspects(text)

    if not aspects:
        return []

    sentences = split_sentences(text)
    results = []

    for aspect in aspects:
        best_context = text  

        for sentence in sentences:
            if aspect_exists(sentence, aspect):
                best_context = extract_aspect_context(sentence, aspect)
                break  

        sentiment_result = get_sentiment(best_context)

        results.append({
            "aspect": aspect,
            "context": best_context,
            "sentiment": sentiment_result["sentiment"],
            "polarity": sentiment_result["polarity"]
        })

    return results


if __name__ == "__main__":

    reviews = [

        "Amazing graphics",
        "Terrible servers",
        "Great gameplay",
        "Boring story",
        "Beautiful graphics but terrible servers",
        "The gameplay is amazing but the story is boring",
        "I love the graphics but I hate the controls",
        "The visuals are stunning and the soundtrack is amazing. "
        "However, the controls feel clunky and the performance needs "
        "serious optimization."
    ]

    for review in reviews:

        print("\n" + "=" * 60)
        print("Review:", review)
        print("=" * 60)

        results = analyze_aspects(review)

        for result in results:
            print(result)