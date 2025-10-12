from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    tokenize_text,
)
from .inverted_index import InvertedIndex


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    # movies = load_movies()
    inv_idx = InvertedIndex()
    inv_idx.load()

    results = []
    query_tokens = tokenize_text(query)
    for token in query_tokens:
        docs_id = inv_idx.get_document(token)
        for doc_id in docs_id:
            doc = inv_idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
