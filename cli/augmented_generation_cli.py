import argparse

from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citation_command,
    question_command,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize the results",
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarizing"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="no. of results returned",
    )

    citation_parser = subparsers.add_parser("citations", help="")
    citation_parser.add_argument("query", type=str, help="Search query for summarizing")
    citation_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="no. of results returned",
    )

    question_parser = subparsers.add_parser("question", help="")
    question_parser.add_argument("question", type=str, help="")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="no. of results returned",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            rag_command(query)
        case "summarize":
            query, limit = args.query, args.limit
            summarize_command(query, limit)
        case "citations":
            query, limit = args.query, args.limit
            citation_command(query, limit)
        case "question":
            question, limit = args.question, args.limit
            question_command(question, limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
