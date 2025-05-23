import argparse
import os
import pytest
import tempfile
from unittest.mock import patch, mock_open, Mock, MagicMock, call
from io import StringIO

from src.main import parse_args, save_tokens_to_file, train_and_evaluate, main


class TestParseArgs:
    """Test class for parse_args function."""

    def test_parse_args_required_corpus(self):
        """Test that corpus argument is required."""
        with patch('sys.argv', ['main.py']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_corpus_only(self):
        """Test parsing with only corpus argument."""
        with patch('sys.argv', ['main.py', '--corpus', 'test_corpus.txt']):
            args = parse_args()
            assert args.corpus == 'test_corpus.txt'
            assert args.alpha == 1.0  # Default value
            assert args.override_models == False  # Default value

    def test_parse_args_all_arguments(self):
        """Test parsing with all arguments."""
        with patch('sys.argv', ['main.py', '--corpus', 'test.txt', '--alpha', '0.5', '--override-models']):
            args = parse_args()
            assert args.corpus == 'test.txt'
            assert args.alpha == 0.5
            assert args.override_models == True

    def test_parse_args_alpha_short_form(self):
        """Test parsing with alpha short form argument."""
        with patch('sys.argv', ['main.py', '--corpus', 'test.txt', '-a', '2.0']):
            args = parse_args()
            assert args.alpha == 2.0

    def test_parse_args_override_short_form(self):
        """Test parsing with override short form argument."""
        with patch('sys.argv', ['main.py', '--corpus', 'test.txt', '-o']):
            args = parse_args()
            assert args.override_models == True


class TestSaveTokensToFileMain:
    """Test class for save_tokens_to_file function in main module."""

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

    @patch('builtins.print')
    def test_save_tokens_to_file_prints_confirmation(self, mock_print, temp_model_dir):
        """Test that function prints confirmation message."""
        tokens = ["test"]
        output_path = os.path.join(temp_model_dir, "tokens.txt")
        
        save_tokens_to_file(tokens, output_path)
        
        mock_print.assert_called_once_with(f"Tokens saved to '{output_path}'.")


class TestTrainAndEvaluate:
    """Test class for train_and_evaluate function."""

    @patch('src.main.compute_perplexity')
    @patch('builtins.print')
    def test_train_and_evaluate_basic(self, mock_print, mock_perplexity, temp_model_dir):
        """Test basic train_and_evaluate functionality."""
        mock_perplexity.return_value = 5.5
        
        train_tokens = ["hello", "world", "test"] * 10
        test_tokens = ["hello", "world"]
        orders = [2, 3]
        
        models = train_and_evaluate(train_tokens, test_tokens, orders, alpha=1.0)
        
        assert len(models) == 2
        assert 2 in models
        assert 3 in models
        
        # Check that perplexity was computed for each model
        assert mock_perplexity.call_count == 2

    @patch('src.main.compute_perplexity')
    @patch('os.path.isfile', return_value=True)
    @patch('os.remove')
    @patch('builtins.print')
    def test_train_and_evaluate_with_override(self, mock_print, mock_remove, mock_isfile, mock_perplexity, temp_model_dir):
        """Test train_and_evaluate with override flag."""
        mock_perplexity.return_value = 3.2
        
        train_tokens = ["hello", "world"] * 5
        test_tokens = ["hello"]
        orders = [2]
        
        models = train_and_evaluate(train_tokens, test_tokens, orders, override=True)
        
        # Should have called os.remove for existing model file
        mock_remove.assert_called()

    @patch('src.main.compute_perplexity')
    @patch('builtins.print')
    def test_train_and_evaluate_custom_alpha(self, mock_print, mock_perplexity, temp_model_dir):
        """Test train_and_evaluate with custom alpha parameter."""
        mock_perplexity.return_value = 2.1
        
        train_tokens = ["a", "b", "c"] * 10
        test_tokens = ["a", "b"]
        orders = [2]
        
        models = train_and_evaluate(train_tokens, test_tokens, orders, alpha=0.5)
        
        assert len(models) == 1
        assert models[2].alpha == 0.5

    @patch('src.main.compute_perplexity')
    @patch('builtins.print')
    def test_train_and_evaluate_prints_perplexity(self, mock_print, mock_perplexity, temp_model_dir):
        """Test that train_and_evaluate prints perplexity for each model."""
        mock_perplexity.side_effect = [1.5, 2.3, 3.1]  # Different perplexities
        
        train_tokens = ["hello", "world"] * 20
        test_tokens = ["hello"]
        orders = [1, 2, 3]
        
        train_and_evaluate(train_tokens, test_tokens, orders)
        
        # Check that perplexity was printed for each order
        expected_calls = [
            call("Perplexity for 1-gram model: 1.5."),
            call("Perplexity for 2-gram model: 2.3."),
            call("Perplexity for 3-gram model: 3.1.")
        ]
        mock_print.assert_has_calls(expected_calls, any_order=True)


class TestMain:
    """Test class for main function."""

    @patch('src.main.parse_args')
    @patch('src.main.read_and_clean_corpus')
    @patch('src.main.save_tokens_to_file')
    @patch('src.main.train_and_evaluate')
    @patch('builtins.input', side_effect=['!q'])  # Quit immediately
    @patch('builtins.print')
    def test_main_success_quit_immediately(self, mock_print, mock_input, mock_train_eval, 
                                          mock_save_tokens, mock_read_corpus, mock_parse_args):
        """Test main function with successful execution and immediate quit."""
        # Setup mocks
        mock_args = Mock()
        mock_args.corpus = "test.txt"
        mock_args.alpha = 1.0
        mock_args.override_models = False
        mock_parse_args.return_value = mock_args
        
        mock_tokens = ["hello", "world"] * 100
        mock_read_corpus.return_value = mock_tokens
        
        mock_models = {1: Mock(), 2: Mock(), 3: Mock(), 4: Mock()}
        mock_train_eval.return_value = mock_models
        
        # Run main
        main()
        
        # Verify calls
        mock_read_corpus.assert_called_once_with(file_path="test.txt")
        mock_save_tokens.assert_called_once()
        mock_train_eval.assert_called_once()

    @patch('src.main.parse_args')
    @patch('src.main.read_and_clean_corpus', side_effect=FileNotFoundError("File not found"))
    @patch('builtins.print')
    def test_main_file_not_found_error(self, mock_print, mock_read_corpus, mock_parse_args):
        """Test main function with file not found error."""
        mock_args = Mock()
        mock_args.corpus = "nonexistent.txt"
        mock_parse_args.return_value = mock_args
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1

    @patch('src.main.parse_args')
    @patch('src.main.read_and_clean_corpus', side_effect=ValueError("Not enough tokens"))
    @patch('builtins.print')
    def test_main_value_error(self, mock_print, mock_read_corpus, mock_parse_args):
        """Test main function with value error."""
        mock_args = Mock()
        mock_args.corpus = "small_file.txt"
        mock_parse_args.return_value = mock_args
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1

    @patch('src.main.parse_args')
    @patch('src.main.read_and_clean_corpus')
    @patch('src.main.save_tokens_to_file')
    @patch('src.main.train_and_evaluate')
    @patch('builtins.input', side_effect=['hello', '!q'])
    @patch('builtins.print')
    def test_main_interactive_prediction(self, mock_print, mock_input, mock_train_eval, 
                                        mock_save_tokens, mock_read_corpus, mock_parse_args):
        """Test main function interactive prediction."""
        # Setup mocks
        mock_args = Mock()
        mock_args.corpus = "test.txt"
        mock_args.alpha = 1.0
        mock_args.override_models = False
        mock_parse_args.return_value = mock_args
        
        mock_tokens = ["hello", "world"] * 100
        mock_read_corpus.return_value = mock_tokens
        
        # Mock models
        mock_model = Mock()
        mock_model.predict_next.return_value = "world"
        mock_models = {1: mock_model, 2: Mock(), 3: Mock(), 4: Mock()}
        mock_train_eval.return_value = mock_models
        
        # Run main
        main()
        
        # Should have prediction print calls
        prediction_calls = [call for call in mock_print.call_args_list if 'Predicted next word' in str(call)]
        assert len(prediction_calls) > 0

    @patch('src.main.parse_args')
    @patch('src.main.read_and_clean_corpus')
    @patch('src.main.save_tokens_to_file')
    @patch('src.main.train_and_evaluate')
    @patch('builtins.input', side_effect=['context without model', '!q'])
    @patch('builtins.print')
    def test_main_no_model_for_order(self, mock_print, mock_input, mock_train_eval, 
                                    mock_save_tokens, mock_read_corpus, mock_parse_args):
        """Test main function when no model exists for required order."""
        # Setup mocks
        mock_args = Mock()
        mock_args.corpus = "test.txt"
        mock_args.alpha = 1.0
        mock_args.override_models = False
        mock_parse_args.return_value = mock_args
        
        mock_tokens = ["hello", "world"] * 100
        mock_read_corpus.return_value = mock_tokens
        
        # Only provide some models, missing the 4-gram model
        mock_models = {1: Mock(), 2: Mock(), 3: Mock()}  # Missing 4-gram
        mock_train_eval.return_value = mock_models
        
        # Run main
        main()
        
        # Should print error about missing model
        error_calls = [call for call in mock_print.call_args_list if 'No model for' in str(call)]
        assert len(error_calls) > 0

    @patch('src.main.parse_args')
    @patch('src.main.read_and_clean_corpus')
    @patch('src.main.save_tokens_to_file')
    @patch('src.main.train_and_evaluate')
    @patch('builtins.input', side_effect=['hello', '!q'])
    @patch('builtins.print')
    def test_main_prediction_exception(self, mock_print, mock_input, mock_train_eval, 
                                      mock_save_tokens, mock_read_corpus, mock_parse_args):
        """Test main function when prediction raises an exception."""
        # Setup mocks
        mock_args = Mock()
        mock_args.corpus = "test.txt"
        mock_args.alpha = 1.0
        mock_args.override_models = False
        mock_parse_args.return_value = mock_args
        
        mock_tokens = ["hello", "world"] * 100
        mock_read_corpus.return_value = mock_tokens
        
        # Mock model that raises exception - need to mock the right model (2-gram for 1 word context)
        mock_model_1 = Mock()
        mock_model_2 = Mock()
        mock_model_2.predict_next.side_effect = Exception("Prediction error")
        mock_models = {1: mock_model_1, 2: mock_model_2, 3: Mock(), 4: Mock()}
        mock_train_eval.return_value = mock_models
        
        # Run main
        main()
        
        # Should print prediction error
        error_calls = [call for call in mock_print.call_args_list if 'Prediction error' in str(call)]
        assert len(error_calls) > 0 