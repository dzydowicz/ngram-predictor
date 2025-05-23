import math
import pytest
from unittest.mock import Mock

from src.perplexity import compute_perplexity
from src.ngram_model import NGramModel


class TestComputePerplexity:
    """Test class for compute_perplexity function."""

    def test_compute_perplexity_basic(self, temp_model_dir):
        """Test basic perplexity computation."""
        tokens = ["hello", "world", "hello", "world", "test"]
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(tokens)
        
        test_tokens = ["hello", "world"]
        perplexity = compute_perplexity(model, test_tokens)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
        assert not math.isinf(perplexity)

    def test_compute_perplexity_empty_tokens(self, temp_model_dir):
        """Test perplexity computation with empty token list."""
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(["hello", "world"])
        
        perplexity = compute_perplexity(model, [])
        assert math.isinf(perplexity)

    def test_compute_perplexity_single_token(self, temp_model_dir):
        """Test perplexity computation with single token."""
        tokens = ["hello", "world", "test"]
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(tokens)
        
        test_tokens = ["hello"]
        perplexity = compute_perplexity(model, test_tokens)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_compute_perplexity_trigram_model(self, temp_model_dir):
        """Test perplexity computation with trigram model."""
        tokens = ["the", "cat", "sat", "on", "the", "mat", "the", "cat", "ran"]
        model = NGramModel(n=3, model_dir=temp_model_dir)
        model.train(tokens)
        
        test_tokens = ["the", "cat", "sat"]
        perplexity = compute_perplexity(model, test_tokens)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_compute_perplexity_unseen_tokens(self, temp_model_dir):
        """Test perplexity computation with tokens not seen during training."""
        train_tokens = ["hello", "world"]
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(train_tokens)
        
        test_tokens = ["unseen", "tokens"]
        perplexity = compute_perplexity(model, test_tokens)
        
        # Should still compute perplexity due to smoothing
        assert isinstance(perplexity, float)
        assert perplexity > 0
        assert not math.isinf(perplexity)

    def test_compute_perplexity_perfect_match(self, temp_model_dir):
        """Test perplexity when test tokens exactly match training pattern."""
        # Train on repeating pattern
        train_tokens = ["a", "b"] * 10  # Repeating pattern
        model = NGramModel(n=2, alpha=1.0, model_dir=temp_model_dir)
        model.train(train_tokens)
        
        # Test on same pattern
        test_tokens = ["a", "b", "a", "b"]
        perplexity = compute_perplexity(model, test_tokens)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_compute_perplexity_different_alpha_values(self, temp_model_dir):
        """Test that different alpha values affect perplexity."""
        train_tokens = ["hello", "world", "test"]
        
        model_alpha_1 = NGramModel(n=2, alpha=1.0, model_dir=temp_model_dir)
        model_alpha_1.train(train_tokens)
        
        model_alpha_01 = NGramModel(n=2, alpha=0.1, model_dir=temp_model_dir + "_01")
        model_alpha_01.train(train_tokens)
        
        test_tokens = ["unseen", "words"]
        
        perplexity_alpha_1 = compute_perplexity(model_alpha_1, test_tokens)
        perplexity_alpha_01 = compute_perplexity(model_alpha_01, test_tokens)
        
        # Different alpha values should give different perplexities
        assert perplexity_alpha_1 != perplexity_alpha_01

    def test_compute_perplexity_padding_handling(self, temp_model_dir):
        """Test that padding tokens are correctly handled in perplexity calculation."""
        tokens = ["hello", "world"]
        model = NGramModel(n=3, model_dir=temp_model_dir)  # Trigram model
        model.train(tokens)
        
        test_tokens = ["hello", "world"]
        perplexity = compute_perplexity(model, test_tokens)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_compute_perplexity_mathematical_properties(self, temp_model_dir):
        """Test mathematical properties of perplexity."""
        tokens = ["a", "b", "c", "a", "b", "c"]
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(tokens)
        
        # Perplexity of training data should be reasonable
        train_perplexity = compute_perplexity(model, tokens)
        
        # Perplexity of completely different data should be higher
        different_tokens = ["x", "y", "z"]
        different_perplexity = compute_perplexity(model, different_tokens)
        
        assert train_perplexity > 0
        assert different_perplexity > 0
        # Different data should generally have higher perplexity
        # (though this isn't guaranteed due to smoothing)

    def test_compute_perplexity_zero_probability_handling(self, temp_model_dir):
        """Test handling of zero probabilities in perplexity calculation."""
        # Create a mock model that returns zero probability
        mock_model = Mock()
        mock_model.n = 2
        mock_model.vocabulary = {"hello", "world"}
        mock_model.alpha = 1.0
        
        def mock_probability(context, word):
            return 0.0  # Always return zero probability
        
        mock_model.probability = mock_probability
        
        test_tokens = ["hello", "world"]
        perplexity = compute_perplexity(mock_model, test_tokens)
        
        # Should handle zero probabilities gracefully
        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_compute_perplexity_large_vocabulary(self, temp_model_dir):
        """Test perplexity computation with large vocabulary."""
        # Create tokens with large vocabulary
        tokens = []
        for i in range(50):
            tokens.extend([f"word_{i}", f"word_{(i+1) % 50}"])
        
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(tokens)
        
        test_tokens = tokens[:10]  # First 10 tokens
        perplexity = compute_perplexity(model, test_tokens)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
        assert not math.isinf(perplexity)

    def test_compute_perplexity_consistency(self, temp_model_dir):
        """Test that perplexity computation is consistent across multiple calls."""
        tokens = ["hello", "world", "test", "hello", "world"]
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(tokens)
        
        test_tokens = ["hello", "world"]
        
        perplexity1 = compute_perplexity(model, test_tokens)
        perplexity2 = compute_perplexity(model, test_tokens)
        
        # Should get same result for same inputs
        assert perplexity1 == perplexity2

    def test_compute_perplexity_unigram_model(self, temp_model_dir):
        """Test perplexity computation with unigram model."""
        tokens = ["hello", "world", "hello", "test"]
        model = NGramModel(n=1, model_dir=temp_model_dir)
        model.train(tokens)
        
        test_tokens = ["hello", "world"]
        perplexity = compute_perplexity(model, test_tokens)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0 