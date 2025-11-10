import argparse

from lib.multimodel_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multi model")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="verify image embedding"
    )
    verify_image_embedding_parser.add_argument("image", type=str, help="image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image = args.image
            verify_image_embedding(image)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
