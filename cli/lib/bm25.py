from .inverted_index import InvertedIndex


def bm25_idf_command(term: str) -> float:
    inv = InvertedIndex()
    inv.load()
    return inv.get_bm25_idf(term)
