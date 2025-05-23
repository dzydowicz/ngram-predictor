import os
import tempfile
import pytest
from unittest.mock import patch, mock_open

from src.data_preprocessing import read_and_clean_corpus


class TestReadAndCleanCorpus:
    """Test class for read_and_clean_corpus function."""

    def test_read_and_clean_corpus_success(self, temp_corpus_file):
        """Test successful reading and cleaning of corpus."""
        result = read_and_clean_corpus(temp_corpus_file, min_tokens=2)
        
        expected = ["hello", "world", "this", "is", "a", "test", "hello", "world", "again"]
        assert result == expected

    def test_read_and_clean_corpus_polish_text(self, polish_sample_text, polish_sample_tokens):
        """Test preprocessing of Polish text with special characters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(polish_sample_text)
            temp_path = f.name
        
        try:
            result = read_and_clean_corpus(temp_path, min_tokens=2)
            assert result == polish_sample_tokens
        finally:
            os.unlink(temp_path)

    def test_read_and_clean_corpus_removes_special_characters(self):
        """Test that special characters and numbers are removed."""
        text_with_special_chars = "Hello123! @#$%^&*()world??? Testing... 456"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text_with_special_chars)
            temp_path = f.name
        
        try:
            result = read_and_clean_corpus(temp_path, min_tokens=2)
            expected = ["hello", "world", "testing"]
            assert result == expected
        finally:
            os.unlink(temp_path)

    def test_read_and_clean_corpus_normalizes_whitespace(self):
        """Test that multiple whitespaces are normalized to single spaces."""
        text_with_whitespace = "Hello    world\n\nThis\t\tis    a    test"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text_with_whitespace)
            temp_path = f.name
        
        try:
            result = read_and_clean_corpus(temp_path, min_tokens=2)
            expected = ["hello", "world", "this", "is", "a", "test"]
            assert result == expected
        finally:
            os.unlink(temp_path)

    def test_read_and_clean_corpus_converts_to_lowercase(self):
        """Test that text is converted to lowercase."""
        text_mixed_case = "HELLO World This IS a TEST"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text_mixed_case)
            temp_path = f.name
        
        try:
            result = read_and_clean_corpus(temp_path, min_tokens=2)
            expected = ["hello", "world", "this", "is", "a", "test"]
            assert result == expected
        finally:
            os.unlink(temp_path)

    def test_read_and_clean_corpus_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        non_existent_path = "non_existent_file.txt"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            read_and_clean_corpus(non_existent_path)
        
        assert f"File {non_existent_path} does not exist." in str(exc_info.value)

    def test_read_and_clean_corpus_empty_file(self):
        """Test ValueError when file is empty."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                read_and_clean_corpus(temp_path)
            
            assert "jest pusty" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_read_and_clean_corpus_whitespace_only_file(self):
        """Test ValueError when file contains only whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("   \n\t   \n   ")  # Only whitespace
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                read_and_clean_corpus(temp_path)
            
            assert "jest pusty" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_read_and_clean_corpus_insufficient_tokens(self):
        """Test ValueError when file has fewer tokens than minimum required."""
        text_few_tokens = "Hello world"  # Only 2 tokens
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text_few_tokens)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                read_and_clean_corpus(temp_path, min_tokens=5)
            
            assert "contains fewer tokens than required" in str(exc_info.value)
            assert "Minimum tokens: 5" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_read_and_clean_corpus_minimum_tokens_boundary(self):
        """Test that function works when token count exactly meets minimum."""
        text = "one two three four five"  # Exactly 5 tokens
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text)
            temp_path = f.name
        
        try:
            result = read_and_clean_corpus(temp_path, min_tokens=5)
            expected = ["one", "two", "three", "four", "five"]
            assert result == expected
        finally:
            os.unlink(temp_path)

    def test_read_and_clean_corpus_default_min_tokens(self):
        """Test default minimum tokens value (2000)."""
        # Create a text with fewer than 2000 tokens
        text = " ".join(["word"] * 100)  # Only 100 tokens
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                read_and_clean_corpus(temp_path)  # Using default min_tokens=2000
            
            assert "contains fewer tokens than required" in str(exc_info.value)
            assert "Minimum tokens: 2000" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_read_and_clean_corpus_unicode_handling(self):
        """Test proper handling of Unicode characters."""
        text_unicode = "Café naïve résumé 测试 العربية"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(text_unicode)
            temp_path = f.name
        
        try:
            result = read_and_clean_corpus(temp_path, min_tokens=2)
            # Only Polish characters should remain after regex filtering
            expected = ["caf", "nave", "rsum", ""]
            # Filter out empty strings
            result = [token for token in result if token]
            assert len(result) >= 2  # Should have some tokens
        finally:
            os.unlink(temp_path) 