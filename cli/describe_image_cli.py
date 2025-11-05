import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=str, help="path to the image file", nargs="?")
    parser.add_argument("--query", type=str, help="query", nargs="?")

    args = parser.parse_args()
    image = args.image
    query = args.query

    multi_model_search(query, image)


def multi_model_search(query: str, image: str):
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"

    with open(image, "rb") as f:
        img = f.read()

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    parts = [
        prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        query.strip(),
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=parts,
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
