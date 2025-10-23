import os

from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = []
        for b in bm25_results:
            bm25_scores.append(b["score"])
        semantic_scores = []
        for s in semantic_results:
            semantic_scores.append(s["score"])

        bm25_normalize = normalize_score(bm25_scores)
        semantic_normalize = normalize_score(semantic_scores)

        document_mapping = {}
        for i, normalized_score in enumerate(bm25_normalize):
            doc = bm25_results[i]
            doc_id = bm25_results[i]["id"]
            if doc_id not in document_mapping:
                document_mapping[doc_id] = {
                    "document": doc,
                    "keyword_score": normalized_score,
                    "semantic_score": 0.0,
                    "hybrid_score": 0.0,
                }

        for i, normalized_score in enumerate(semantic_normalize):
            doc = semantic_results[i]
            doc_id = semantic_results[i]["id"]
            if doc_id not in document_mapping:
                document_mapping[doc_id] = {
                    "document": doc,
                    "keyword_score": 0.0,
                    "semantic_score": normalized_score,
                    "hybrid_score": 0.0,
                }
            else:
                document_mapping[doc_id]["semantic_score"] = normalized_score

        for k, v in document_mapping.items():
            bm25_score = v["keyword_score"]
            semantic_score = v["semantic_score"]
            document_mapping[k]["hybrid_score"] = hybrid_score(
                bm25_score, semantic_score, alpha
            )

        return sorted(
            document_mapping.items(),
            key=lambda x: x[1]["hybrid_score"],
            reverse=True,
        )

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_score(scores: list[float]):
    if len(scores) == 0:
        return []

    max_value = max(scores)
    min_value = min(scores)
    res_scores = []

    if max_value == min_value:
        res_scores = [1.0] * len(scores)
        return res_scores

    for i in scores:
        normalized_value = (i - min_value) / (max_value - min_value)
        res_scores.append(normalized_value)

    return res_scores


def weighted_search_command(query: str, alpha: float = 0.5, limit: int = 5):
    movies = load_movies()
    hybrid_searcher = HybridSearch(movies)
    results = hybrid_searcher.weighted_search(query, alpha, limit)
    for i, res in enumerate(results[:limit], 1):
        print(f"{i}. {res[1]["document"]["title"]}")
        print(f"     Hybrid Score: {res[1]["hybrid_score"]}")
        print(
            f"     BM25: {res[1]["keyword_score"]}, Semantic: {res[1]["semantic_score"]}"
        )
