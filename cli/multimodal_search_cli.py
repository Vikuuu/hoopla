import argparse

from lib.multimodel_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Multi model")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="verify image embedding"
    )
    verify_image_embedding_parser.add_argument("image", type=str, help="image path")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movie by providing the image path"
    )
    image_search_parser.add_argument("image", type=str, help="image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image = args.image
            verify_image_embedding(image)
        case "image_search":
            image = args.image
            image_search_command(image)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
