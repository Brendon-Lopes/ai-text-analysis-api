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


def summarize_text(text: str, num_sentences: int = 3) -> str:
    """
    Summarizes the text by selecting the most relevant sentences.

    Uses a frequency-based algorithm:
    1. Counts how often each meaningful word appears in the full text.
    2. Scores each sentence by the average frequency of its words.
    3. Returns the top N sentences in their original order.

    Note: this is a baseline, extractive summarization approach.
    """
    doc = nlp(text)
    sentences = list(doc.sents)

    if len(sentences) <= num_sentences:
        return text

    # Count word frequencies across the full text (ignoring stopwords/punctuation)
    word_frequencies: dict[str, int] = {}
    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            word = token.text.lower()
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    # Score each sentence by the average frequency of its words
    sentence_scores: dict[int, float] = {}
    for index, sentence in enumerate(sentences):
        words_in_sentence = [
            token.text.lower()
            for token in sentence
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        if words_in_sentence:
            total_frequency = sum(word_frequencies.get(word, 0) for word in words_in_sentence)
            sentence_scores[index] = total_frequency / len(words_in_sentence)

    # Pick the top N sentences and sort them back into their original order
    top_indices = sorted(
        sorted(sentence_scores, key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
    )

    return " ".join(sentences[i].text.strip() for i in top_indices)
