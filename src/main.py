import os
import sys
from src.data_preprocessing import read_and_clean_corpus


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
        return

    file_path = sys.argv[1]
    output_path = os.path.join("data", "tokens_output.txt")

    try:
        tokens = read_and_clean_corpus(file_path=file_path)
        print(f"Number of tokens: {len(tokens)}")
        print(f"Sample tokens: {tokens[:40]}")
        save_tokens_to_file(tokens, output_path)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
