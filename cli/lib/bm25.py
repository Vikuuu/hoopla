from .inverted_index import InvertedIndex
from .search_utils import BM25_K1, BM25_B


def bm25_idf_command(term: str) -> float:
    inv = InvertedIndex()
    inv.load()
    return inv.get_bm25_idf(term)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    inv = InvertedIndex()
    inv.load()
    return inv.get_bm25_tf(doc_id, term, k1)
