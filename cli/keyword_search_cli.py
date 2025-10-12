#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser(
        "build", help="Build the Inverted Index on the movie data"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            res = search_command(args.query)
            for i, m in enumerate(res):
                print(f"{i}. {m["title"]}")
        case "build":
            inv_idx = InvertedIndex()
            inv_idx.build()
            inv_idx.save()
            docs = inv_idx.get_document("merida")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
