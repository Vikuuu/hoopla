import os

from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.__index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError(
            "Weighted hybrid search is not \
            implemented yet."
        )

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_score(scores: list[float]):
    if len(scores) == 0:
        return

    max_value = max(scores)
    min_value = min(scores)
    res_scores = []

    if max_value == min_value:
        res_scores = [1.0] * len(scores)
        print(res_scores)
        return

    for i in scores:
        normalized_value = (i - min_value) / (max_value - min_value)
        res_scores.append(normalized_value)

    for s in res_scores:
        print(f"* {s:.4f}")
