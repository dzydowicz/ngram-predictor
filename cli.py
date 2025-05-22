import argparse
from typing import Optional

from src.service import NGramService, DEFAULT_ORDERS


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    :return: Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and interact with N-gram language models."
    )
    parser.add_argument(
        "--corpus",
        required = True,
        help = "Path to the corpus file."
    )
    parser.add_argument(
        '--alpha', '-a',
        type = float,
        default = 1.0,
        help = 'Laplace smoothing parameter (default: 1.0)'
    )
    parser.add_argument(
        '--override-models', '-o',
        action = 'store_true',
        help = 'Force retraining by overriding existing saved models'
    )
    return parser.parse_args()

def interactive_loop(svc: NGramService) -> None:
    print(f"----------------------------------------------------------")
    print(f"Enter up to {svc.max_context} words as context or '!q' to quit.")
    while True:
        user_input: str = input(f"Context> ").strip()

        if user_input.lower() == "!q":
            print("Quitting.")
            break

        context_tokens: list[str] = user_input.lower().split()
        m = len(context_tokens)

        if m > svc.max_context:
            print(f"Error: Too many words ({m}). Max context length is {svc.max_context}.")
            continue

        try:
            next_word: Optional[str] = svc.predict_next(tuple(context_tokens))
            print(f"Predicted next word: '{next_word}'.")
        except Exception as e:
            print(f"Error: {e}")

def main() -> None:
    args = parse_args()

    svc = NGramService(
        corpus_path=args.corpus,
        orders=DEFAULT_ORDERS,
        alpha=args.alpha,
        override_models=args.override_models,
    )

    interactive_loop(svc)

if __name__ == "__main__":
    main()