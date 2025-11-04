import os
import time
import json

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder


from .keyword_search import InvertedIndex
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch

load_dotenv()


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(
            bm25_results,
            semantic_results,
            alpha,
        )
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(
            query,
            limit * 500,
        )

        combined = combine_rrf_search_results(
            bm25_results,
            semantic_results,
            k,
        )
        return combined[:limit]


def rank_search_results(results: list[dict], k: int = DEFAULT_K) -> list[dict]:
    scores: list[int] = []
    for rank, result in enumerate(results, 1):
        scores.append(rrf_score(rank, k))

    for i, result in enumerate(results):
        result["rrf_score"] = scores[i]

    return results


def combine_rrf_search_results(
    bm25_results: list[dict], semantic_results: list[dict], k: int = DEFAULT_K
) -> list[dict]:
    bm25_ranked = rank_search_results(bm25_results, k)
    semantic_ranked = rank_search_results(semantic_results, k)

    combined_scores = {}

    for result in bm25_ranked:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": result["rrf_score"],
                "semantic_rank": 0.0,
            }
    for result in semantic_ranked:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": 0.0,
                "semantic_rank": result["rrf_score"],
            }
        else:
            combined_scores[doc_id]["semantic_rank"] = result["rrf_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        rank_value = data["bm25_rank"] + data["semantic_rank"]
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=rank_value,
            bm25_score=data["bm25_rank"],
            semantic_score=data["semantic_rank"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def rrf_score(rank: int, k: int = DEFAULT_K):
    return 1 / (k + rank)


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))

    return normalized_scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict],
    semantic_results: list[dict],
    alpha: float = DEFAULT_ALPHA,
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(
            data["bm25_score"],
            data["semantic_score"],
            alpha,
        )
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def rrf_search_command(
    query: str,
    k: int = DEFAULT_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance: str = "",
    rerank: str = "",
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query
    print(f"Original Query: {original_query}")
    if enhance is not None or enhance != "":
        query = update_query(original_query, enhance)
        print(f"Query after enhancement: {query}")

    search_limit = limit
    if rerank == "individual":
        search_limit *= 5

    results = searcher.rrf_search(query, k, search_limit)
    print("The results")
    for res in results:
        print(f"Title: {res["title"]}")
        print(f"Score: {res["score"]}")
        print(f"semantic: {res["metadata"]["semantic_score"]:.4f}")
        print(f"keyword: {res["metadata"]["bm25_score"]:.4f}")
    if rerank != "":
        results = rerank_method(query, results, rerank)
        print("The results afte re-ranking")
        for res in results:
            print(f"Title: {res["title"]}")
            print(f"Score: {res["new_score"]}")

    return {
        "original_query": original_query,
        "query": query,
        "k": k,
        "results": results[:limit],
    }


def rerank_method(query: str, documents: list[dict], method: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    if method == "individual":
        for doc in documents:
            genai_prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
            print("sleeping for 10 sec")
            time.sleep(10)

            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=genai_prompt,
            )
            new_score = response.text.strip()
            doc["new_score"] = new_score

        return sorted(documents, key=lambda x: x["new_score"], reverse=True)

    elif method == "batch":
        genai_prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{documents}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
        """
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=genai_prompt,
        )
        new_rank_order = json.loads(response.text.strip())
        print(documents)
        for doc in documents:
            doc_id = doc["id"]
            rank = new_rank_order.index(doc_id)
            doc["new_score"] = rank + 1

        return sorted(documents, key=lambda x: x["new_score"], reverse=True)
    elif method == "cross_encoder":
        pairs = []
        for doc in documents:
            pairs.append(
                [
                    query,
                    f"{doc.get('title', '')} - {doc.get('description', '')}",
                ]
            )
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        scores = cross_encoder.predict(pairs)
        for i, doc in enumerate(documents):
            doc["new_score"] = scores[i]

        return sorted(documents, key=lambda x: x["new_score"], reverse=True)


def update_query(query: str, method: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    if method == "" or method is None:
        return query

    if method == "spell":
        genai_query = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    elif method == "rewrite":
        genai_query = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    elif method == "expand":
        genai_query = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=genai_query,
    )

    enhanced_query = response.text.strip()
    print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")

    return enhanced_query


def evaluate_results(query: str, results: dict):
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    formatted_results = []
    for i, r in enumerate(results, 1):
        formatted_results.append(
            f"{i}. {r['title']} (score: {r['score']})" f"  {r['document']}"
        )

    genai_query = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=genai_query,
    )
    rankings = json.loads(response.text.strip())
    for i, res in enumerate(results):
        print(f"{i+1}. {res['title']}: {rankings[i]}/3")
