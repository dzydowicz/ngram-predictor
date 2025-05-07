import os
import sys
from src.data_preprocessing import read_and_clean_corpus
from src.ngram_model import NGramModel


def save_tokens_to_file(tokens, output_path):
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_text_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_path = os.path.join("data", "tokens_output.txt")

    # 1. Read and clean corpus
    try:
        tokens = read_and_clean_corpus(file_path=file_path)
        save_tokens_to_file(tokens, output_path)
        print(f"Tokens preprocessed, number of tokens: {len(tokens)}.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Train models for n = 1,2,3,4
    default_orders = [1, 2, 3, 4]
    alpha = 1.0
    models = {}

    for n in default_orders:
        model = NGramModel(n, alpha)
        model.train(tokens)
        models[n] = model

    print(f"Models trained for orders: {default_orders}.")

    max_ctx = max(default_orders) - 1

    print(f"Enter up to {max_ctx} words as context or '!exit' to quit.")
    while True:
        # 3. Get and validate user input
        user_input = input(f"Context> ").strip()

        if user_input.lower() == "!exit":
            print("Quitting.")
            break

        context_tokens = user_input.lower().split()
        m = len(context_tokens)

        if m > max_ctx:
            print(f"Error: Too many words ({m}). Max context length is {max_ctx}.")
            continue

        # 4. Pick a model
        order = m + 1
        model = models.get(order)
        if model is None:
            print(f"Error: No model for {order}.")
            continue

        # 5. Predict next word
        try:
            next_word = model.predict_next(tuple(context_tokens))
            print(f"Predicted next word: '{next_word}'.")
        except Exception as e:
            print(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
