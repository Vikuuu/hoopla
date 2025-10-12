import pickle
import os

from collections import Counter, defaultdict

from .search_utils import (
    tokenize_text,
    load_movies,
    PROJECT_ROOT,
)


class InvertedIndex:
    def __init__(self):
        self.attribute: dict[str, set] = {}
        self.docmap: dict[int, str] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)

        self.__index_path = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
        self.__docmap_path = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")
        self.__term_freq_path = os.path.join(
            PROJECT_ROOT, "cache", "term_frequencies.pkl"
        )

    def __add_document(self, doc_id: int, text: str):
        tokens: list[str] = tokenize_text(text)
        for token in tokens:
            if token not in self.attribute:
                value = set()
                value.add(doc_id)
                self.attribute[token] = value
                continue

            self.attribute[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)

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
        self.save()

    def save(self):
        index_data = pickle.dumps(self.attribute)
        docmap_data = pickle.dumps(self.docmap)
        term_freq_data = pickle.dumps(self.term_frequencies)
        os.mkdir(os.path.join(PROJECT_ROOT, "cache"))

        with open(self.__index_path, "wb") as f:
            f.write(index_data)
        with open(self.__docmap_path, "wb") as f:
            f.write(docmap_data)
        with open(self.__term_freq_path, "wb") as f:
            f.write(term_freq_data)

    def load(self):
        try:
            with open(self.__index_path, "rb") as f:
                self.attribute = pickle.load(f)
            with open(self.__docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.__term_freq_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except Exception as e:
            print(f"{e}")

    def get_tf(self, doc_id: str, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise Exception("Len to tokens greater that 1")
        return self.term_frequencies[int(doc_id)][tokens[0]]
