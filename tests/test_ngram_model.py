import os
import pickle
import tempfile
import pytest
from collections import defaultdict
from unittest.mock import patch, mock_open, MagicMock

from src.ngram_model import NGramModel


class TestNGramModel:
    """Test class for NGramModel."""

    def test_init_default_parameters(self, temp_model_dir):
        """Test NGramModel initialization with default parameters."""
        model = NGramModel(n=2, model_dir=temp_model_dir)
        
        assert model.n == 2
        assert model.alpha == 1.0
        assert isinstance(model.ngram_counts, defaultdict)
        assert isinstance(model.context_counts, defaultdict)
        assert isinstance(model.vocabulary, set)
        assert len(model.vocabulary) == 0
        assert model.model_path.endswith("ngram_2gram_model.pkl")

    def test_init_custom_parameters(self, temp_model_dir):
        """Test NGramModel initialization with custom parameters."""
        model = NGramModel(n=3, alpha=0.5, model_dir=temp_model_dir)
        
        assert model.n == 3
        assert model.alpha == 0.5
        assert model.model_path.endswith("ngram_3gram_model.pkl")

    def test_train_new_model(self, temp_model_dir, sample_tokens):
        """Test training a new model from scratch."""
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(sample_tokens)
        
        # Check vocabulary
        assert model.vocabulary == set(sample_tokens)
        
        # Check some expected bigrams
        assert model.ngram_counts[("<s>", "hello")] > 0
        assert model.ngram_counts[("hello", "world")] > 0
        assert model.ngram_counts[("world", "this")] > 0
        
        # Check context counts
        assert model.context_counts[("<s>",)] > 0
        assert model.context_counts[("hello",)] > 0

    def test_train_with_existing_model_file(self, temp_model_dir, sample_tokens):
        """Test that training loads existing model if file exists."""
        # First, create and train a model
        model1 = NGramModel(n=2, model_dir=temp_model_dir)
        model1.train(sample_tokens)
        original_vocab_size = len(model1.vocabulary)
        
        # Create a new model instance with same parameters
        model2 = NGramModel(n=2, model_dir=temp_model_dir)
        model2.train(["new", "tokens"])  # Different tokens
        
        # Should load the existing model, not train on new tokens
        assert len(model2.vocabulary) == original_vocab_size
        assert model2.vocabulary == set(sample_tokens)

    def test_predict_next_valid_context(self, temp_model_dir):
        """Test prediction with valid context."""
        tokens = ["hello", "world", "hello", "world", "hello", "test"]
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(tokens)
        
        # After "hello", both "world" and "test" are possible, but "world" is more likely
        prediction = model.predict_next(("hello",))
        assert prediction in ["world", "test"]

    def test_predict_next_trigram(self, temp_model_dir):
        """Test prediction with trigram model."""
        tokens = ["the", "cat", "sat", "the", "cat", "ran", "the", "dog", "sat"]
        model = NGramModel(n=3, model_dir=temp_model_dir)
        model.train(tokens)
        
        prediction = model.predict_next(("the", "cat"))
        assert prediction in ["sat", "ran"]

    def test_predict_next_invalid_context_length(self, temp_model_dir, sample_tokens):
        """Test prediction with invalid context length."""
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(sample_tokens)
        
        # Too many words in context
        with pytest.raises(ValueError) as exc_info:
            model.predict_next(("too", "many", "words"))
        assert "Content length 3 is not 1" in str(exc_info.value)
        
        # Too few words in context
        with pytest.raises(ValueError) as exc_info:
            model.predict_next(())
        assert "Content length 0 is not 1" in str(exc_info.value)

    def test_predict_next_no_matches(self, temp_model_dir):
        """Test prediction when no matching context is found."""
        tokens = ["hello", "world"]
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(tokens)
        
        # Context not seen during training
        prediction = model.predict_next(("unknown",))
        # Should return some word from vocabulary due to smoothing
        assert prediction is None or prediction in model.vocabulary

    def test_probability_calculation(self, temp_model_dir):
        """Test probability calculation with Laplace smoothing."""
        tokens = ["hello", "world", "hello", "world"]
        model = NGramModel(n=2, alpha=1.0, model_dir=temp_model_dir)
        model.train(tokens)
        
        # P("world" | "hello") should be high
        prob_world = model.probability(("hello",), "world")
        
        # P("test" | "hello") should be low but non-zero due to smoothing
        prob_test = model.probability(("hello",), "test")
        
        assert prob_world > prob_test
        assert prob_test > 0  # Should be non-zero due to smoothing
        assert 0 <= prob_world <= 1
        assert 0 <= prob_test <= 1

    def test_probability_unseen_context(self, temp_model_dir, sample_tokens):
        """Test probability calculation for unseen context."""
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(sample_tokens)
        
        # Unseen context should still give non-zero probability due to smoothing
        prob = model.probability(("unseen",), "word")
        assert prob > 0

    def test_probability_invalid_context_length(self, temp_model_dir, sample_tokens):
        """Test probability calculation with invalid context length."""
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(sample_tokens)
        
        with pytest.raises(ValueError) as exc_info:
            model.probability(("too", "many"), "word")
        assert "Content length 2 is not 1" in str(exc_info.value)

    def test_save_and_load_model(self, temp_model_dir, sample_tokens):
        """Test saving and loading model functionality."""
        # Create and train model
        model1 = NGramModel(n=2, alpha=0.5, model_dir=temp_model_dir)
        model1.train(sample_tokens)
        
        # Load the saved model
        model2 = NGramModel._load(model1.model_path)
        
        # Check that loaded model has same parameters and data
        assert model2.n == model1.n
        assert model2.alpha == model1.alpha
        assert model2.vocabulary == model1.vocabulary
        assert dict(model2.ngram_counts) == dict(model1.ngram_counts)
        assert dict(model2.context_counts) == dict(model1.context_counts)

    def test_laplace_smoothing_alpha_parameter(self, temp_model_dir):
        """Test that different alpha values affect probability calculations."""
        tokens = ["a", "b", "a", "b"]
        
        model_alpha_1 = NGramModel(n=2, alpha=1.0, model_dir=temp_model_dir)
        model_alpha_1.train(tokens)
        
        model_alpha_01 = NGramModel(n=2, alpha=0.1, model_dir=temp_model_dir + "_01")
        model_alpha_01.train(tokens)
        
        # For unseen word, lower alpha should give lower probability
        prob_alpha_1 = model_alpha_1.probability(("a",), "unseen")
        prob_alpha_01 = model_alpha_01.probability(("a",), "unseen")
        
        assert prob_alpha_01 < prob_alpha_1

    def test_padding_tokens(self, temp_model_dir):
        """Test that padding tokens are correctly added."""
        tokens = ["hello", "world"]
        model = NGramModel(n=3, model_dir=temp_model_dir)
        model.train(tokens)
        
        # Should have trigrams starting with padding tokens
        assert model.ngram_counts[("<s>", "<s>", "hello")] == 1
        assert model.ngram_counts[("<s>", "hello", "world")] == 1

    def test_empty_token_list(self, temp_model_dir):
        """Test training with empty token list."""
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train([])
        
        assert len(model.vocabulary) == 0
        assert len(model.ngram_counts) == 0
        assert len(model.context_counts) == 0

    def test_single_token_list(self, temp_model_dir):
        """Test training with single token."""
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(["hello"])
        
        assert model.vocabulary == {"hello"}
        assert model.ngram_counts[("<s>", "hello")] == 1
        assert model.context_counts[("<s>",)] == 1

    def test_model_path_generation(self, temp_model_dir):
        """Test that model path is correctly generated."""
        model = NGramModel(n=4, model_dir=temp_model_dir)
        expected_filename = "ngram_4gram_model.pkl"
        assert model.model_path.endswith(expected_filename)
        assert temp_model_dir in model.model_path

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    def test_save_method(self, mock_pickle_dump, mock_file, temp_model_dir):
        """Test the _save method."""
        model = NGramModel(n=2, alpha=0.5, model_dir=temp_model_dir)
        model.vocabulary = {"test", "words"}
        model.ngram_counts[("test", "words")] = 1
        model.context_counts[("test",)] = 1
        
        model._save()
        
        # Check that file was opened for writing
        mock_file.assert_called_once_with(model.model_path, "wb")
        
        # Check that pickle.dump was called with correct data
        mock_pickle_dump.assert_called_once()
        saved_data = mock_pickle_dump.call_args[0][0]
        
        assert saved_data["n"] == 2
        assert saved_data["alpha"] == 0.5
        # Since vocabulary is a set, the order in the list can vary
        assert set(saved_data["vocabulary"]) == {"test", "words"}

    def test_large_vocabulary_performance(self, temp_model_dir):
        """Test model performance with larger vocabulary."""
        # Create tokens with larger vocabulary
        tokens = []
        for i in range(100):
            tokens.extend([f"word_{i}", f"word_{(i+1) % 100}"])
        
        model = NGramModel(n=2, model_dir=temp_model_dir)
        model.train(tokens)
        
        assert len(model.vocabulary) == 100
        
        # Test prediction works
        prediction = model.predict_next((f"word_0",))
        assert prediction is not None

    def test_probability_edge_cases(self, temp_model_dir):
        """Test probability calculation edge cases."""
        model = NGramModel(n=2, alpha=1.0, model_dir=temp_model_dir)
        model.train(["a", "b"])
        
        # Test with very small vocabulary
        prob = model.probability(("a",), "b")
        assert 0 < prob <= 1
        
        # Test probability sums (approximately) to 1 for all possible words
        total_prob = sum(model.probability(("a",), word) for word in model.vocabulary)
        assert abs(total_prob - 1.0) < 0.1  # Allow for floating point errors 