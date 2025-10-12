import pickle
import os

from .search_utils import (
    tokenize_text,
    load_movies,
    PROJECT_ROOT,
)


class InvertedIndex:
    def __init__(self):
        self.attribute: dict[str, set] = {}
        self.docmap: dict = {}

        self.__index_path = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
        self.__docmap_path = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")

    def __add_document(self, doc_id: int, text: str):
        tokens: list[str] = tokenize_text(text)
        for token in tokens:
            if token not in self.attribute:
                value = set()
                value.add(doc_id)
                self.attribute[token] = value
                continue

            self.attribute[token].add(doc_id)

    def get_document(self, term: str):
        results = []
        if term.lower() in self.attribute:
            results = self.attribute[term.lower()]
        return sorted(list(results))

    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_id = movie["id"]
            assert movie_id

            text = f"{movie["title"]} {movie["description"]}"
            self.__add_document(movie_id, text)
            self.docmap[movie_id] = movie

    def save(self):
        index_data = pickle.dumps(self.attribute)
        docmap_data = pickle.dumps(self.docmap)
        os.mkdir(os.path.join(PROJECT_ROOT, "cache"))

        with open(self.__index_path, "wb") as f:
            f.write(index_data)
        with open(self.__docmap_path, "wb") as f:
            f.write(docmap_data)

    def load(self):
        try:
            with open(self.__index_path, "rb") as f:
                self.attribute = pickle.load(f)
            with open(self.__docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        except Exception as e:
            print(f"{e}")


class Document: ...
