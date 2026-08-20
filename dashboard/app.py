from flask import Flask, render_template, request

import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.bert.predict_bert import predict

from aspect_analysis.aspect_extraction import (
    extract_aspects
)

from aspect_analysis.aspect_sentiment import (
    analyze_aspects
)

app = Flask(__name__)


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    review = request.form.get(
        "review",
        ""
    ).strip()

    # Check empty review
    if not review:

        return render_template(
            "result.html",
            review="",
            sentiment="neutral",
            confidence=0,
            extracted_aspects=[],
            aspect_results=[]
        )

    # ------------------------------------
    # 1. Overall sentiment using BERT
    # ------------------------------------

    sentiment_result = predict(
        review
    )

    # ------------------------------------
    # 2. Extract aspects
    # ------------------------------------

    extracted_aspects = extract_aspects(
        review
    )

    # ------------------------------------
    # 3. Analyze sentiment of each aspect
    # ------------------------------------

    aspect_results = analyze_aspects(
        review
    )

    # ------------------------------------
    # 4. Send everything to template
    # ------------------------------------

    return render_template(

        "result.html",

        review=review,

        # Overall sentiment
        sentiment=sentiment_result[
            "sentiment"
        ],

        confidence=round(
            sentiment_result[
                "confidence"
            ] * 100,
            2
        ),

        # Aspect names
        extracted_aspects=extracted_aspects,

        # Aspect + context + sentiment
        aspect_results=aspect_results
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )