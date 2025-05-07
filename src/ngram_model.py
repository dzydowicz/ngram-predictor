
import os
import pickle
from collections import defaultdict


class NGramModel:
    """
    N-gram model with Laplace smoothing.
    """

    def __init__(self, n:int, alpha=1.0, model_dir="data/processed"):
        """
        :param n: the order of the n-gram.
        :param alpha: Laplace smoothing parameter.
        :param model_dir: directory for saving/loading the model.
        """
        self.n = n
        self.alpha = alpha
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()

        os.makedirs(model_dir, exist_ok=True)
        filename = f"ngram_{self.n}gram_model.pkl"
        self.model_path = os.path.join(model_dir, filename)

    @classmethod
    def _load(cls, path):
        """
        Loads the model from a pickle file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(n=data["n"], alpha=data["alpha"], model_dir=os.path.dirname(path))
        model.ngram_counts = defaultdict(int, data["ngram_counts"])
        model.context_counts = defaultdict(int, data["context_counts"])
        model.vocabulary = set(data["vocabulary"])
        return model

    def train(self, tokens):
        """
        If the file at self.model_path exists, loads the model from this file.
        Otherwise, computes n-grams, updates dictionary, and saves the model.
        :param tokens: list of preprocessed tokens.
        """
        if os.path.isfile(self.model_path):
            loaded = self._load(self.model_path)
            self.ngram_counts = loaded.ngram_counts
            self.context_counts = loaded.context_counts
            self.vocabulary = loaded.vocabulary
            return

        self.vocabulary.update(tokens)
        pad = ["<s>"] * (self.n - 1)
        seq = pad + tokens

        for i in range(len(seq) - self.n + 1):
            gram = tuple(seq[i : i + self.n])
            ctx = gram[:-1]
            self.ngram_counts[gram] += 1
            self.context_counts[ctx] += 1

        self._save()

    def predict_next(self, context):
        if len (context) != self.n - 1:
            raise ValueError(f"Content length {len(context)} is not {self.n - 1}.")

        best_word, best_prob = None, 0.0

        for w in self.vocabulary:
            p = self.probability(context, w)
            if p > best_prob:
                best_word = w
                best_prob = float(p)

        return best_word

    def _probability(self, context, word):
        if len(context) != self.n - 1:
            raise ValueError(f"Content length {len(context)} is not {self.n - 1}.")

        count_ng = self.ngram_counts[context + (word,)]
        count_ctx = self.context_counts[context]
        V = len(self.vocabulary)

        num = count_ng + self.alpha
        den = count_ctx + self.alpha * V
        return num / den if den > 0 else 0.0

    def _save(self):
        """
        Saves the model to a pickle file.
        """
        data = {
            "n": self.n,
            "alpha": self.alpha,
            "ngram_counts": dict(self.ngram_counts),
            "context_counts": dict(self.context_counts),
            "vocabulary": list(self.vocabulary)
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)
