import os
import tempfile
from unittest.mock import patch

import pytest


@pytest.fixture
def sample_tokens():
    """Fixture providing sample tokens for testing."""
    return ["hello", "world", "this", "is", "a", "test", "hello", "world", "again"]


@pytest.fixture
def sample_text():
    """Fixture providing sample text data."""
    return "Hello world! This is a test. Hello world again."


@pytest.fixture
def temp_corpus_file(sample_text):
    """Fixture that creates a temporary corpus file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(sample_text)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_model_dir():
    """Fixture that creates a temporary directory for model files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model_dir():
    """Fixture that mocks the model directory to prevent file I/O during tests."""
    with patch('os.makedirs'), \
         patch('os.path.exists', return_value=False), \
         patch('os.path.isfile', return_value=False):
        yield


@pytest.fixture
def polish_sample_text():
    """Fixture providing Polish sample text for preprocessing tests."""
    return "Witaj świecie! To jest test. Ąćęłńóśźż - polskie znaki."


@pytest.fixture
def polish_sample_tokens():
    """Fixture providing expected Polish tokens after preprocessing."""
    return ["witaj", "świecie", "to", "jest", "test", "ąćęłńóśźż", "polskie", "znaki"] 