import argparse
import os
import sys
from typing import Optional

from src.data_preprocessing import read_and_clean_corpus
from src.ngram_model import NGramModel
from src.perplexity import compute_perplexity


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

def save_tokens_to_file(tokens: list[str], output_path: str):
    """
    Saves a list of tokens to a text file, one per line.

    :param tokens: List of string tokens.
    :param output_path: Path where the file should be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(f"{token}\n")
    print(f"Tokens saved to '{output_path}'.")

def train_and_evaluate(
        train_tokens: list[str],
        test_tokens: list[str],
        orders: list[int],
        alpha: float = 1.0,
        override: bool = False
) -> dict[int, NGramModel]:
   """
   Trains NGram models for specified orders, evaluates perplexity.

   :param train_tokens: List of tokens to train on.
   :param test_tokens: List of tokens to test on.
   :param orders: List of n-gram orders.
   :param override: If true, existing models are retrained regardless of saved files.
   """
   models: dict[int, NGramModel] = {}

   for n in orders:
       model: NGramModel = NGramModel(n, alpha)

       if override and os.path.isfile(model.model_path):
           os.remove(model.model_path)

       model.train(train_tokens)
       models[n] = model

       perplexity: float = compute_perplexity(model, test_tokens)
       print(f"Perplexity for {n}-gram model: {perplexity}.")

   return models

def main():
    args = parse_args()

    file_path: str = args.corpus
    output_path = os.path.join("data", "tokens_output.txt")

    # 1. Read and clean corpus
    try:
        tokens = read_and_clean_corpus(file_path=file_path)
        save_tokens_to_file(tokens, output_path)
        print(f"Tokens preprocessed, number of tokens: {len(tokens)}.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    split_ratio: float = 0.8
    split_idx: int = int(len(tokens) * split_ratio)
    train_tokens: list[str] = tokens[:split_idx]
    test_tokens: list[str] = tokens[split_idx:]
    print(f"Training on {len(train_tokens)} tokens, testing on {len(test_tokens)} tokens.")

    # 2. Train models for n = 1,2,3,4
    default_orders: list[int] = [1, 2, 3, 4]
    models: dict[int, NGramModel] = train_and_evaluate(train_tokens,
                                                       test_tokens,
                                                       default_orders,
                                                       alpha=args.alpha,
                                                       override=args.override_models)

    max_ctx = max(default_orders) - 1

    print(f"----------------------------------------------------------")
    print(f"Enter up to {max_ctx} words as context or '!q' to quit.")
    while True:
        # 3. Get and validate user input
        user_input: str = input(f"Context> ").strip()

        if user_input.lower() == "!q":
            print("Quitting.")
            break

        context_tokens: list[str] = user_input.lower().split()
        m = len(context_tokens)

        if m > max_ctx:
            print(f"Error: Too many words ({m}). Max context length is {max_ctx}.")
            continue

        # 4. Pick a model
        order: int = m + 1
        model: Optional[NGramModel] = models.get(order)
        if model is None:
            print(f"Error: No model for {order}.")
            continue

        # 5. Predict next word
        try:
            next_word: Optional[str] = model.predict_next(tuple(context_tokens))
            print(f"Predicted next word: '{next_word}'.")
        except Exception as e:
            print(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
