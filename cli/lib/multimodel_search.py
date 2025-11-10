from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import load_movies


class MultiModal:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
        )

    def embed_image(self, image_path: str):
        image_data = Image.open(image_path)
        return self.model.encode(image_data)

    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)
        cosine_similarities = []
        for idx, embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(
                embedding,
                image_embedding,
            )
            doc = self.documents[idx]
            cosine_similarities.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "score": score,
                }
            )

        return sorted(
            cosine_similarities,
            key=lambda x: x["score"],
            reverse=True,
        )[:5]


def verify_image_embedding(image_path: str):
    multi_model = MultiModal()
    embedding = multi_model.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(img_path):
    movies = load_movies()
    multi_model = MultiModal(movies)
    results = multi_model.search_with_image(img_path)

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (similarity: {res['score']:.3f})")
        print(f"     {res['description'][:100]}")
