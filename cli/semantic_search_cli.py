#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the installed model.")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Embeds the given text"
    )
    _ = embed_text_parser.add_argument("text", type=str, help="text to be embedded")

    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Load or Creates the embeddings"
    )

    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Create embedding for the search query"
    )
    _ = embedquery_parser.add_argument("text", type=str, help="Query text")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic searching"
    )
    _ = search_parser.add_argument("query", type=str, help="Query text")
    _ = search_parser.add_argument(
        "--limit",
        type=int,
        # nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit the search return",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk size")
    _ = chunk_parser.add_argument("text", type=str, help="text to chunk")
    _ = chunk_parser.add_argument(
        "--chunk-size", nargs="?", type=int, default=200, help="chuck size"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.text)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunked_text = args.text.split()
            print(f"Chunking {len(args.text)} characters")
            curr = []
            x = 1
            for idx, i in enumerate(chunked_text):
                curr.append(i)
                if len(curr) == args.chunk_size or idx == len(chunked_text) - 1:
                    print(f"{x}. {" ".join(curr)}")
                    curr = []
                    x += 1
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
