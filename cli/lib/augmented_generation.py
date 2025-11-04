import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import rrf_search_command, HybridSearch
from .search_utils import load_movies

load_dotenv()


def rag_command(query: str):
    docs = rrf_search_command(query=query, limit=5)

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs['results']}

Provide a comprehensive answer that addresses the query:"""

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )

    print("Search Results:")
    for i, doc in enumerate(docs["results"], 1):
        print(f"  - {doc['title']}")

    print("RAG Response:")
    print(f"  {response.text.strip()}")


def summarize_command(query: str, limit: int = 5):
    movies = load_movies()
    searcher = HybridSearch(movies)

    results = searcher.rrf_search(query=query, k=50, limit=limit)
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{'\n'.join(res['title'] for res in results)}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
    print(len(prompt))

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )

    print("Search Results:")
    for i, doc in enumerate(results, 1):
        print(f"  - {doc['title']}")

    print("RAG Response:")
    print(f"  {response.text.strip()}")
