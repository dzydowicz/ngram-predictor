import math

from src.ngram_model import NGramModel

def compute_perplexity(model: NGramModel, tokens) -> float:
    """
    Computes the perplexity score for a given n-gram model, based on given tokens.
    """
    pad = ['<s>'] * (model.n - 1)
    seq = pad + tokens
    N = len(tokens)

    if N == 0:
        return float('inf')

    log_prob_sum: float = 0.0
    V: int = len(model.vocabulary)

    for i in range(len(seq) - model.n + 1):
        context = tuple(seq[i:i + model.n - 1])
        word = seq[i + model.n - 1]
        p = model.probability(context, word)

        if p > 0:
            log_prob_sum += math.log(p)
        else:
            log_prob_sum += math.log(model.alpha / (model.alpha * V))

    perplexity = math.exp(-log_prob_sum / N)
    return perplexity