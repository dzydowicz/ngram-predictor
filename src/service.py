import os
import sys
from typing import Sequence, List, Dict, Optional

from src.data_preprocessing import read_and_clean_corpus
from src.ngram_model import NGramModel
from src.perplexity import compute_perplexity

DEFAULT_ORDERS = [1, 2, 3, 4]

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

class NGramService:
    """
    Service layer for NGram models management.
    """
    def __init__(
            self,
            corpus_path: str,
            orders: Sequence[int] = DEFAULT_ORDERS,
            alpha: float = 1.0,
            *,
            override_models: bool = False,
            train_ratio: float = 0.8,
    ) -> None:
        self.corpus_path = corpus_path
        self.orders = list(orders)
        self.alpha = alpha
        self.override = override_models
        self.train_ratio = train_ratio

        self.tokens: List[str] = []
        self.models: Dict[int, NGramModel] = {}
        self.perplexities: Dict[str, float] = {}

        self._startup()

    @property
    def max_context(self) -> int:
        return max(self.orders) - 1

    def predict_next(self, context: Sequence[str]) -> Optional[str]:
        m = len(context)

        if m > self.max_context:
            raise ValueError(f"Too many words ({m}). Max context length is {self.max_context}.")

        order = m + 1
        model = self.models.get(order)

        if model is None:
            raise RuntimeError(f"No model for {order}.")

        return model.predict_next(tuple(map(str.lower, context)))
        
    def get_model_stats(self) -> Dict:
        """
        Returns model statistics including stored perplexity values.
        """
        return {
            "perplexity": self.perplexities,
            "vocabulary_size": len(self.tokens),
            "alpha": self.alpha
        }

    def _startup(self) -> None:
        output_path = os.path.join("data", "tokens_output.txt")

        try:
            self.tokens = read_and_clean_corpus(file_path=self.corpus_path)
            save_tokens_to_file(self.tokens, output_path)
            print(f"Tokens preprocessed, number of tokens: {len(self.tokens)}.")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

        split_idx = int(len(self.tokens) * self.train_ratio)
        train_tokens = self.tokens[:split_idx]
        test_tokens = self.tokens[split_idx:]
        print(f"Training on {len(train_tokens)} tokens, testing on {len(test_tokens)} tokens.")

        self.models = self._train_and_evaluate(
            train_tokens,
            test_tokens
        )

    def _train_and_evaluate(
            self,
            train_tokens: list[str],
            test_tokens: list[str]
    ) -> dict[int, NGramModel]:
        """
        Trains NGram models for specified orders, evaluates perplexity.

        :param train_tokens: List of tokens to train on.
        :param test_tokens: List of tokens to test on.
        """
        models: dict[int, NGramModel] = {}

        for n in self.orders:
            model: NGramModel = NGramModel(n, self.alpha)

            if self.override and os.path.isfile(model.model_path):
                os.remove(model.model_path)

            model.train(train_tokens)
            models[n] = model

            perplexity: float = compute_perplexity(model, test_tokens)
            self.perplexities[f"{n}-gram"] = round(perplexity, 2)
            print(f"Perplexity for {n}-gram model: {perplexity}.")

        return models
