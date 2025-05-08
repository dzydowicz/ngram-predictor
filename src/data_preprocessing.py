import os
import re

"""
Module responsible for reading and pre-processing text corpus.
It provides functions for validation, cleaning and tokenizing the text.
"""

def read_and_clean_corpus(file_path: str, min_tokens: int = 2000) -> list[str]:
    """
    Reads a text file and preprocesses it to produce a list of tokens.

    :param file_path: Path to the corpus file.
    :param min_tokens: Minimum number of tokens required for proper processing.
    :return: A list of tokens (words) cleaned of unnecessary characters.
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file is empty or has fewer tokens than required.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as file:
        text: str = file.read()

    if not text.strip():
        raise ValueError(f"Plik '{file_path}' jest pusty.")

    text = text.lower()
    text = re.sub(r'[^a-ząćęłńóśźż\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens: list[str] = text.split()

    if len(tokens) < min_tokens:
        raise ValueError(
            f"File '{file_path}' contains fewer tokens than required. Minimum tokens: {min_tokens}."
        )

    return tokens