import re
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from aspect_analysis.aspect_labels import ASPECTS

def extract_aspects(text):
    if not isinstance(text, str):
        return []

    text = text.lower()

    found_aspects = []

    for aspect, keywords in ASPECTS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                found_aspects.append(aspect)
                break

    return found_aspects

def extract_aspects_with_keywords(text):
    if not isinstance(text, str):
        return []

    text_lower = text.lower()

    result = {}

    for aspect, keywords in ASPECTS.items():
        matched_keywords = []
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        if matched_keywords:
            result[aspect] = matched_keywords

    return result

if __name__ == "__main__":
    review_text = "The graphics are amazing and the gameplay or combat is fun, but the servers have terrible lag."
    print("Extracted aspects:", extract_aspects(review_text))
    print("Extracted aspects with keywords:", extract_aspects_with_keywords(review_text))