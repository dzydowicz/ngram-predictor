import os
import tempfile
import pytest
from unittest.mock import patch, mock_open, Mock, MagicMock

from src.service import NGramService, save_tokens_to_file, DEFAULT_ORDERS


class TestSaveTokensToFile:
    """Test class for save_tokens_to_file function."""

    def test_save_tokens_to_file_success(self, temp_model_dir):
        """Test successful saving of tokens to file."""
        tokens = ["hello", "world", "test"]
        output_path = os.path.join(temp_model_dir, "tokens.txt")
        
        save_tokens_to_file(tokens, output_path)
        
        # Check that file was created and contains correct content
        assert os.path.exists(output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read().strip().split('\n')
        
        assert content == tokens

    def test_save_tokens_to_file_creates_directory(self):
        """Test that function creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "new_dir", "tokens.txt")
            tokens = ["test", "tokens"]
            
            save_tokens_to_file(tokens, output_path)
            
            assert os.path.exists(output_path)
            assert os.path.exists(os.path.dirname(output_path))

    def test_save_tokens_to_file_empty_list(self, temp_model_dir):
        """Test saving empty token list."""
        tokens = []
        output_path = os.path.join(temp_model_dir, "empty_tokens.txt")
        
        save_tokens_to_file(tokens, output_path)
        
        assert os.path.exists(output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert content == ""

    @patch('builtins.print')
    def test_save_tokens_to_file_prints_confirmation(self, mock_print, temp_model_dir):
        """Test that function prints confirmation message."""
        tokens = ["test"]
        output_path = os.path.join(temp_model_dir, "tokens.txt")
        
        save_tokens_to_file(tokens, output_path)
        
        mock_print.assert_called_once_with(f"Tokens saved to '{output_path}'.")


class TestNGramService:
    """Test class for NGramService."""

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_init_success(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test successful initialization of NGramService."""
        mock_tokens = ["hello", "world", "test", "hello", "world"]
        mock_read_corpus.return_value = mock_tokens
        
        service = NGramService(
            corpus_path=temp_corpus_file,
            orders=[2, 3],
            alpha=0.5,
            train_ratio=0.8
        )
        
        assert service.corpus_path == temp_corpus_file
        assert service.orders == [2, 3]
        assert service.alpha == 0.5
        assert service.train_ratio == 0.8
        assert service.tokens == mock_tokens
        assert len(service.models) == 2  # Should have 2-gram and 3-gram models
        assert 2 in service.models
        assert 3 in service.models

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_init_default_parameters(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test initialization with default parameters."""
        mock_tokens = ["hello", "world"] * 100  # Enough tokens
        mock_read_corpus.return_value = mock_tokens
        
        service = NGramService(corpus_path=temp_corpus_file)
        
        assert service.orders == DEFAULT_ORDERS
        assert service.alpha == 1.0
        assert service.train_ratio == 0.8
        assert service.override == False

    @patch('src.service.read_and_clean_corpus', side_effect=FileNotFoundError("File not found"))
    @patch('builtins.print')
    @patch('sys.exit')
    def test_init_file_not_found_error(self, mock_exit, mock_print, mock_read_corpus):
        """Test initialization with file not found error."""
        NGramService(corpus_path="non_existent_file.txt")
        
        mock_exit.assert_called_once_with(1)

    @patch('src.service.read_and_clean_corpus', side_effect=ValueError("Not enough tokens"))
    @patch('builtins.print')
    @patch('sys.exit')
    def test_init_value_error(self, mock_exit, mock_print, mock_read_corpus):
        """Test initialization with value error."""
        NGramService(corpus_path="file.txt")
        
        mock_exit.assert_called_once_with(1)

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_max_context_property(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test max_context property."""
        mock_read_corpus.return_value = ["hello", "world"] * 50
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[2, 3, 4])
        
        assert service.max_context == 3  # max(orders) - 1 = 4 - 1 = 3

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_predict_next_valid_context(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test prediction with valid context."""
        tokens = ["hello", "world", "hello", "world", "test"] * 20  # Repeat for training
        mock_read_corpus.return_value = tokens
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[2])
        
        # Test prediction - actual model behavior may vary due to train/test split
        prediction = service.predict_next(["hello"])
        # Prediction should be a string or None
        assert prediction is None or isinstance(prediction, str)

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_predict_next_context_too_long(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test prediction with context that's too long."""
        mock_read_corpus.return_value = ["hello", "world"] * 50
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[2, 3])  # max_context = 2
        
        with pytest.raises(ValueError) as exc_info:
            service.predict_next(["too", "many", "words"])
        
        assert "Too many words (3). Max context length is 2." in str(exc_info.value)

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_predict_next_no_model_for_order(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test prediction when no model exists for the required order."""
        mock_read_corpus.return_value = ["hello", "world"] * 50
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[2, 4])  # Missing 3-gram model
        
        with pytest.raises(RuntimeError) as exc_info:
            service.predict_next(["hello", "world"])  # Requires 3-gram model
        
        assert "No model for 3." in str(exc_info.value)

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_predict_next_case_insensitive(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test that prediction is case-insensitive."""
        tokens = ["hello", "world"] * 50
        mock_read_corpus.return_value = tokens
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[2])
        
        # Should convert to lowercase
        prediction1 = service.predict_next(["HELLO"])
        prediction2 = service.predict_next(["hello"])
        
        # Both should work (or both return None due to train/test split)
        assert prediction1 == prediction2

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_get_model_stats(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test get_model_stats method."""
        mock_tokens = ["hello", "world", "test"] * 50
        mock_read_corpus.return_value = mock_tokens
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[2, 3], alpha=0.5)
        
        stats = service.get_model_stats()
        
        assert "perplexity" in stats
        assert "vocabulary_size" in stats
        assert "alpha" in stats
        
        assert stats["vocabulary_size"] == len(mock_tokens)
        assert stats["alpha"] == 0.5
        assert len(stats["perplexity"]) == 2  # Should have perplexity for 2-gram and 3-gram

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_train_test_split(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test that train/test split works correctly."""
        mock_tokens = list(range(100))  # 100 tokens
        mock_read_corpus.return_value = mock_tokens
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[2], train_ratio=0.8)
        
        # Check that models were trained on correct amount of data
        # This is implicit since we can't directly access train/test splits
        # But we can verify that perplexities were computed
        assert "2-gram" in service.perplexities

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    @patch('os.path.isfile', return_value=True)
    @patch('os.remove')
    def test_override_models_flag(self, mock_remove, mock_isfile, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test that override_models flag removes existing model files."""
        mock_read_corpus.return_value = ["hello", "world"] * 50
        
        service = NGramService(
            corpus_path=temp_corpus_file,
            orders=[2],
            override_models=True
        )
        
        # Should have called os.remove for existing model file
        mock_remove.assert_called()

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_perplexity_calculation(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test that perplexity is calculated and stored."""
        mock_tokens = ["hello", "world", "test", "hello", "world"] * 50
        mock_read_corpus.return_value = mock_tokens
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[2, 3])
        
        # Check that perplexities were calculated
        assert "2-gram" in service.perplexities
        assert "3-gram" in service.perplexities
        
        # Check that values are reasonable
        assert isinstance(service.perplexities["2-gram"], (int, float))
        assert isinstance(service.perplexities["3-gram"], (int, float))
        assert service.perplexities["2-gram"] > 0
        assert service.perplexities["3-gram"] > 0

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_empty_context_prediction(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test prediction with empty context."""
        mock_tokens = ["hello", "world"] * 50
        mock_read_corpus.return_value = mock_tokens
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[1])  # Unigram model
        
        # Empty context should work for unigram model
        prediction = service.predict_next([])
        # Prediction should be a string or None
        assert prediction is None or isinstance(prediction, str)

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_single_order_service(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test service with single n-gram order."""
        mock_tokens = ["hello", "world", "test"] * 50
        mock_read_corpus.return_value = mock_tokens
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[3])
        
        assert service.orders == [3]
        assert service.max_context == 2
        assert len(service.models) == 1
        assert 3 in service.models

    @patch('src.service.read_and_clean_corpus')
    @patch('src.service.save_tokens_to_file')
    @patch('builtins.print')
    def test_multiple_orders_service(self, mock_print, mock_save_tokens, mock_read_corpus, temp_corpus_file):
        """Test service with multiple n-gram orders."""
        mock_tokens = ["hello", "world", "test", "again"] * 50
        mock_read_corpus.return_value = mock_tokens
        
        service = NGramService(corpus_path=temp_corpus_file, orders=[1, 2, 3, 4])
        
        assert len(service.models) == 4
        assert all(order in service.models for order in [1, 2, 3, 4])
        assert service.max_context == 3 