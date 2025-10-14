#!/usr/bin/env python3

import argparse
import math

from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex
from lib.bm25 import bm25_idf_command, bm25_tf_command
from lib.search_utils import BM25_K1, BM25_B


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
    _ = idf_parser.add_argument("term", type=str, help="Term for which idf is needed")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Returns the Term Frequest-Inverse Docuement Frequecy number"
    )
    _ = tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    _ = tfidf_parser.add_argument(
        "term", type=str, help="Term for which tf-idf is needed"
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    _ = bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )

    _ = bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    _ = bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for"
    )
    _ = bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    _ = bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    _ = bm25search_parser.add_argument("query", type=str, help="Search query")
    _ = bm25search_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="Limit the resulting values"
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
        case "tfidf":
            inv_idx = InvertedIndex()
            inv_idx.load()
            tf = inv_idx.get_tf(args.doc_id, args.term)
            docs = search_command(args.term, len(inv_idx.docmap))
            idf = math.log((len(inv_idx.docmap) + 1) / (len(docs) + 1))
            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            inv = InvertedIndex()
            inv.load()
            results = inv.bm25_search(args.query, args.limit)
            for count, result in enumerate(results, 1):
                print(
                    f"{count}. ({result["id"]}) {result["title"]} - Score: {result["score"]}"
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
