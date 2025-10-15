import numpy as np
import os

from sentence_transformers import SentenceTransformer

from .search_utils import PROJECT_ROOT, load_movies


class SemanticSearch:
    np_file_path = os.path.join(PROJECT_ROOT, "cache", "movie_embeddings.npy")

    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents: list[dict] = None
        self.document_map: dict = {}

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("Given empty text")
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embedding(self, documents: list[str]):
        self.documents = documents
        self.document_map = {}
        movie_string = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_string.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_string, show_progress_bar=True)
        os.makedirs(os.path.dirname(self.np_file_path), exist_ok=True)
        np.save(self.np_file_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[str]):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.np_file_path):
            self.embeddings = np.load(self.np_file_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embedding(documents)

    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embed = self.generate_embedding(query)

        doc_similarity_scores = []
        for i, embedding in enumerate(self.embeddings, 1):
            similarity_score = cosine_similarity(embedding, query_embed)
            doc_similarity_scores.append((similarity_score, self.document_map[i]))

        doc_similarity_scores = sorted(
            doc_similarity_scores, key=lambda x: x[0], reverse=True
        )
        doc_similarity_scores = doc_similarity_scores[:limit]

        results = []
        for doc in doc_similarity_scores:
            results.append(
                {
                    "score": doc[0],
                    "title": doc[1]["title"],
                    "description": doc[1]["description"],
                }
            )

        return results


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_model():
    s = SemanticSearch()
    print(f"Model loaded: {s.model}")
    print(f"Max sequence length: {s.model.max_seq_length}")


def embed_text(text: str):
    s = SemanticSearch()
    embedding = s.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    s = SemanticSearch()
    documents = load_movies()
    embeddings = s.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    s = SemanticSearch()
    embedding = s.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search(query: str, limit: int):
    s = SemanticSearch()
    documents = load_movies()
    embeddings = s.load_or_create_embeddings(documents)
    results = s.search(query, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res["title"]} (score: {res["score"]})")
        print(f"     {res["description"][:50]}")
