from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

TEXT = "Artificial intelligence is transforming software engineering and the world."

LONG_TEXT = (
    "FastAPI is a modern web framework for building APIs with Python. "
    "It is based on standard Python type hints and provides automatic documentation. "
    "Many developers love FastAPI for its speed and simplicity. "
    "The framework is built on top of Starlette and Pydantic. "
    "FastAPI is one of the fastest Python frameworks available today."
)


def test_keywords_returns_list():
    response = client.post("/text/keywords", json={"text": TEXT})

    assert response.status_code == 200
    data = response.json()
    assert "keywords" in data
    assert isinstance(data["keywords"], list)


def test_keywords_are_strings():
    response = client.post("/text/keywords", json={"text": TEXT})

    keywords = response.json()["keywords"]
    assert all(isinstance(k, str) for k in keywords)


def test_keywords_returns_at_most_ten():
    response = client.post("/text/keywords", json={"text": TEXT})

    keywords = response.json()["keywords"]
    assert len(keywords) <= 10


def test_sentiment_returns_valid_label():
    response = client.post("/text/sentiment", json={"text": "This is a great and wonderful product."})

    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ("positive", "negative", "neutral")


def test_sentiment_returns_score():
    response = client.post("/text/sentiment", json={"text": "This is a great and wonderful product."})

    data = response.json()
    assert "score" in data
    assert -1.0 <= data["score"] <= 1.0


def test_sentiment_positive_text():
    response = client.post("/text/sentiment", json={"text": "Excellent and amazing, truly wonderful!"})

    assert response.json()["sentiment"] == "positive"


def test_sentiment_negative_text():
    response = client.post("/text/sentiment", json={"text": "This is terrible and awful and horrible."})

    assert response.json()["sentiment"] == "negative"


def test_sentiment_handles_negation():
    # "not working very great" should NOT be interpreted as positive
    response = client.post("/text/sentiment", json={"text": "my internet is not working very great today"})

    assert response.json()["sentiment"] != "positive"


def test_summary_returns_string():
    response = client.post("/text/summary", json={"text": LONG_TEXT})

    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert isinstance(data["summary"], str)
    assert len(data["summary"]) > 0


def test_summary_is_shorter_than_original():
    response = client.post("/text/summary", json={"text": LONG_TEXT})

    summary = response.json()["summary"]
    assert len(summary) < len(LONG_TEXT)


def test_endpoints_reject_missing_text():
    for endpoint in ("/text/keywords", "/text/sentiment", "/text/summary"):
        response = client.post(endpoint, json={})
        assert response.status_code == 422
