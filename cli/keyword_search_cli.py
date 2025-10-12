#!/usr/bin/env python3

import argparse
import math

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

    tf_parser = subparsers.add_parser(
        "tf", help="Returns the number of time a term appears in the document"
    )
    _ = tf_parser.add_argument("doc_id", type=str, help="Docuement ID")
    _ = tf_parser.add_argument("term", type=str, help="Term for which count is needed")

    idf_parser = subparsers.add_parser(
        "idf", help="Returns the Inverse Document Frequecy number"
    )
    _ = idf_parser.add_argument("term", type=str, help="Term for which idf in needed")

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
        case "tf":
            inv_idx = InvertedIndex()
            inv_idx.load()
            count = inv_idx.get_tf(args.doc_id, args.term)
            print(count)
        case "idf":
            inv_idx = InvertedIndex()
            inv_idx.load()
            docs = search_command(args.term, len(inv_idx.docmap))
            idf = math.log((len(inv_idx.docmap) + 1) / (len(docs) + 1))
            print(f"Inverse document frequecy of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
