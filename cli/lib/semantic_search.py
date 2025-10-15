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
        for document in documents:
            self.document_map[document["id"]] = document
        embeddings = []
        for document in documents:
            val = f"{document["title"]}: {document["description"]}"
            embeddings.append(val)

        self.embeddings = self.model.encode(embeddings, show_progress_bar=True)
        self.np_file_path = os.path.join(PROJECT_ROOT, "cache", "movie_embeddings.npy")
        np.save(self.np_file_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[str]):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(self.np_file_path):
            self.embeddings = np.load(self.np_file_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embedding(documents)


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
