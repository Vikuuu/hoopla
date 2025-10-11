import json
import os

DEFAULT_SEARCH_LIMIT = 5

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
