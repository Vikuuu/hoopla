import pickle
import os
import math

from collections import Counter, defaultdict

from .search_utils import (
    tokenize_text,
    load_movies,
    PROJECT_ROOT,
    BM25_K1,
    BM25_B,
    format_search_result,
)


class InvertedIndex:
    def __init__(self):
        self.attribute: dict[str, set] = {}
        self.docmap: dict[int, str] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}

        self.__index_path = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
        self.__docmap_path = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")
        self.__term_freq_path = os.path.join(
            PROJECT_ROOT, "cache", "term_frequencies.pkl"
        )
        self.__doc_lenghts_path = os.path.join(PROJECT_ROOT, "cache", "doc_lengths.pkl")

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
        self.doc_lengths[doc_id] = len(tokens)

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

            text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie_id, text)
            self.docmap[movie_id] = movie
        self.save()

    def save(self):
        index_data = pickle.dumps(self.attribute)
        docmap_data = pickle.dumps(self.docmap)
        term_freq_data = pickle.dumps(self.term_frequencies)
        doc_len_data = pickle.dumps(self.doc_lengths)
        os.mkdir(os.path.join(PROJECT_ROOT, "cache"))

        with open(self.__index_path, "wb") as f:
            f.write(index_data)
        with open(self.__docmap_path, "wb") as f:
            f.write(docmap_data)
        with open(self.__term_freq_path, "wb") as f:
            f.write(term_freq_data)
        with open(self.__doc_lenghts_path, "wb") as f:
            f.write(doc_len_data)

    def load(self):
        try:
            with open(self.__index_path, "rb") as f:
                self.attribute = pickle.load(f)
            with open(self.__docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.__term_freq_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.__doc_lenghts_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except Exception as e:
            print(f"{e}")

    def get_tf(self, doc_id: str, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise Exception("Len to tokens greater that 1")
        return self.term_frequencies[int(doc_id)][tokens[0]]

    def get_idf(self, doc_id: int, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise Exception("Len to tokens greater that 1")
        N = len(self.docmap)
        df = len(self.get_document(tokens[0]))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            len_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            len_norm = 1
        tf_component = (tf * (k1 + 1)) / (tf + k1 * len_norm)
        return tf_component

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_len = 0
        for length in self.doc_lengths.values():
            total_len += length
        return total_len / len(self.doc_lengths)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int) -> dict[int, float]:
        query_tokens: list[str] = tokenize_text(query)
        scores: dict[int, float] = {}

        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
