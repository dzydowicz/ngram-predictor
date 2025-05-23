# N-gram Predictor

A Python-based N-gram language model for predicting the next word in a sequence. This project implements statistical language modeling using n-grams with Laplace smoothing and provides both a web interface and command-line interface for interaction.

## Features

- **Multiple N-gram Orders**: Supports unigram, bigram, trigram, and 4-gram models
- **Laplace Smoothing**: Handles unseen word sequences with configurable smoothing parameter
- **Web Interface**: Modern, responsive web UI for easy interaction
- **RESTful API**: FastAPI-based backend with endpoints for prediction and model statistics
- **CLI Interface**: Command-line tool for training and interactive prediction
- **Model Persistence**: Automatic model saving/loading to avoid retraining
- **Perplexity Calculation**: Evaluate model performance with standard metrics

## Sample Data

The project includes a sample corpus (`data/data.txt`) containing Polish text about technology topics including artificial intelligence, programming, data analysis, machine learning, databases, cybersecurity, and computer networks. 

**Data Split**: The corpus is automatically divided during model training - 80% is used for training the n-gram models, and 20% is reserved for validation and perplexity calculation. This split ensures unbiased evaluation of model performance on unseen data.

**Note**: This sample corpus was generated using AI (ChatGPT) to provide diverse training data for demonstration purposes. For production use, consider using real-world text corpora appropriate to your specific domain.

## Installation

### Prerequisites

- Python 3.8+ 
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ngram-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter `python: command not found`, try using `python3` instead:
```bash
python3 -m pip install -r requirements.txt
```

## Usage

### Web Interface

The web interface consists of two separate servers:

**Option 1: Quick Start (Recommended)**
```bash
python start_web.py
# or if python command not found:
python3 start_web.py
```
This convenience script starts both servers automatically and provides clear status information.

**Option 2: Manual Setup**

**1. Start the API backend server:**
```bash
python api.py
# or if python command not found:
python3 api.py
```
This starts the FastAPI backend on `http://localhost:8000` (API endpoints only)

**2. Start the frontend web server:**
```bash
python web_server.py
# or if python command not found:
python3 web_server.py
```
This starts the frontend server on `http://localhost:3000`

**3. Usage:**
- Open your browser and navigate to: `http://localhost:3000`
- Enter 1-3 words as context and get the predicted next word
- The frontend will automatically communicate with the API backend

**Note:** Both servers need to be running simultaneously for the web interface to work properly.

### Command Line Interface

Train and interact with the model via CLI:

```bash
python cli.py --corpus data/data.txt
# or:
python3 cli.py --corpus data/data.txt
```

Options:
- `--corpus`: Path to your text corpus (required)
- `--alpha`: Laplace smoothing parameter (default: 1.0)
- `--override-models`: Force model retraining

### API Endpoints

The FastAPI server provides these endpoints:

- `POST /predict`: Predict next word
  ```json
  {
    "context": ["artificial", "intelligence"]
  }
  ```

- `GET /model-stats`: Get model statistics including perplexity scores

## Technical Details

### N-gram Models

The system implements multiple n-gram orders:
- **Unigram (1-gram)**: Single word probabilities
- **Bigram (2-gram)**: Two-word sequence probabilities  
- **Trigram (3-gram)**: Three-word sequence probabilities
- **4-gram**: Four-word sequence probabilities

### Smoothing

Uses Laplace (add-alpha) smoothing to handle unseen n-grams:
```
P(w|context) = (count(context, w) + α) / (count(context) + α × |V|)
```

### Prediction Strategy

The model uses a backoff strategy, preferring longer contexts when available:
1. Try 4-gram prediction (if 3 words context)
2. Fall back to trigram (if 2+ words context)  
3. Fall back to bigram (if 1+ words context)
4. Fall back to unigram (most frequent word)

## Example Usage

### Python API
```python
from src.service import NGramService

# Initialize service
service = NGramService("data/data.txt")

# Predict next word
next_word = service.predict_next(["sztuczna", "inteligencja"])
print(f"Next word: {next_word}")

# Get model statistics
stats = service.get_model_stats()
print(f"Vocabulary size: {stats['vocabulary_size']}")
```

### cURL API Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"context": ["sztuczna", "inteligencja"]}'
```

## Performance

- Model training time depends on corpus size
- Prediction is typically sub-millisecond for cached models
- Memory usage scales with vocabulary size and n-gram order
- Supports corpora up to several million tokens

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Troubleshooting

### Common Issues

**"python: command not found"**
- Use `python3` instead of `python` on macOS/Linux systems

**Memory issues with large corpora**
- Reduce n-gram orders or increase system memory
- Consider text preprocessing to reduce vocabulary size

**Poor prediction quality** 
- Increase corpus size
- Adjust smoothing parameter (`--alpha`)
- Ensure corpus matches target domain
