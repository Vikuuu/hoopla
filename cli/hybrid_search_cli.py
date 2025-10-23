import argparse

from lib.hybrid_search import normalize_score


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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_score(args.list)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
