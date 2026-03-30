import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()


def extract_keywords(text: str) -> list[str]:
    """
    Extracts meaningful keywords from the text.

    Uses spaCy part-of-speech tagging to keep only nouns and proper nouns,
    removing stopwords and punctuation. Returns up to 10 unique keywords.
    """
    doc = nlp(text)

    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN")
        and not token.is_stop
        and not token.is_punct
        and token.is_alpha
    ]

    # Remove duplicates while preserving order of appearance
    seen: set[str] = set()
    unique_keywords: list[str] = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)

    return unique_keywords[:10]


def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of the text using VADER (Valence Aware Dictionary
    and sEntiment Reasoner).

    VADER is a model trained specifically for sentiment analysis. It understands:
    - Negation:      "not great"     → negative
    - Intensifiers:  "very good"     → more positive
    - Punctuation:   "great!!!"      → stronger signal
    - Capitalization: "GREAT"        → stronger signal

    Returns a compound score between -1.0 (most negative) and 1.0 (most positive),
    and a label: "positive", "negative", or "neutral".

    Thresholds used (standard VADER recommendation):
    - compound >= 0.05  → positive
    - compound <= -0.05 → negative
    - otherwise         → neutral
    """
    scores = analyzer.polarity_scores(text)
    compound = round(scores["compound"], 4)

    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"sentiment": sentiment, "score": compound}

