import json
import os
import string

from typing import Any

from nltk import PorterStemmer


DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 2

DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAPPING = 0
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORD_PATH, "r") as f:
        data = f.read()
        return data.splitlines()


def tokenize_text(text: str) -> list[str]:
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token and token not in stopwords:
            valid_tokens.append(stemmer.stem(token))
    return valid_tokens


def preprocess_text(text: str) -> str:
    text = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
