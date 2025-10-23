import argparse

from lib.hybrid_search import normalize_score, weighted_search_command
from lib.search_utils import HYBRID_ALPHA, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize the provided list of numbers"
    )
    normalize_parser.add_argument(
        "list",
        type=float,
        nargs="+",
        help="List of float values",
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Search using hybrid"
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="Query to search data on."
    )
    weighted_search_parser.add_argument(
        "--alpha",
        nargs="?",
        default=HYBRID_ALPHA,
        type=float,
        help="Tuning which side to give more preference to",
    )
    weighted_search_parser.add_argument(
        "--limit",
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        type=int,
        help="Limit the results returned",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_score(args.list)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
