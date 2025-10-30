import argparse
import unittest
import json

from lib.search_utils import GOLDEN_DATASET_PATH
from lib.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    evaluation(limit)


def evaluation(limit: int):
    with open(GOLDEN_DATASET_PATH, "r") as f:
        golden_dataset = json.load(f)

    golden_dataset = golden_dataset["test_cases"]
    for data in golden_dataset:
        query = data["query"]
        results = rrf_search_command(
            query=query,
            k=60,
            limit=limit,
        )["results"]
        total_retrieved = len(results)
        relevent_retrieved = 0
        for res in results:
            if res["title"] in data["relevant_docs"]:
                relevent_retrieved += 1
        precision = relevent_retrieved / total_retrieved

        print(f"- Query: {data["query"]}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        retrieved = []
        for res in results:
            retrieved.append(res["title"])
        print(f"  - Retrieved: {", ".join(retrieved)}")
        print(f"  - Relevant:  {", ".join(data["relevant_docs"])}")


class TestEvaluation(unittest.TestCase):
    def test_evaluation(self):
        self.assertEqual(evaluation(6), "")


if __name__ == "__main__":
    main()
