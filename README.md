# Code RAG Generator

A Retrieval-Augmented Generation (RAG) system for intelligent code generation using FAISS vector store and the HumanEval dataset. This system finds semantically similar code examples and uses them as context to generate new, high-quality Python functions.

## Features

- **FAISS Vector Store**: Efficient similarity search using cosine similarity with normalized vectors
- **HumanEval Dataset**: Pre-trained on OpenAI's HumanEval coding problems (164 examples)
- **Smart Retrieval**: Finds the most relevant code examples using semantic similarity
- **Code Generation**: Generates Python functions using context from similar examples via OpenRouter API
- **Automated Testing**: Built-in test case validation and code quality assessment
- **Modular Design**: Clean, function-based architecture for easy customization

## Project Structure

```
rag_code_generator/
├── main.py                 # Main pipeline and demo execution
├── config.py              # Configuration settings and environment variables
├── data_loader.py         # HumanEval dataset loading and processing
├── embeddings.py          # Text embedding generation with sentence transformers
├── faiss_index.py         # FAISS vector store management with cosine similarity
├── code_generator.py      # LLM-based code generation with OpenRouter
└── utils.py              # Utility functions, testing, and result visualization
```

## Installation

### Install dependencies

```bash
pip install datasets sentence-transformers faiss-cpu openai python-dotenv torch numpy
```

### Set up environment variables

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your-api-key-here
```

## Requirements

- Python 3.8+
- See `requirements.txt` for detailed dependencies

## Quick Start

### Basic Usage

```python
from main import setup_rag_pipeline, generate_code_for_task

# Set up the complete pipeline
pipeline = setup_rag_pipeline(api_key="your-openrouter-key")

# Generate code for a task
task = """
def calculate_median(numbers: List[float]) -> float:
    '''Calculate the median of a list of numbers.'''
"""

result = generate_code_for_task(pipeline, task)
print(result['generated_code'])
```

### Run Demo

```bash
python main.py
```

The demo automatically tests the system with multiple programming tasks.

## Configuration

Customize the system in `config.py`:

```python
# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
GENERATION_MODEL = "deepseek/deepseek-coder"  # Code generation model

# Search settings
DEFAULT_TOP_K = 3      # Number of similar examples to retrieve
MAX_TOKENS = 500       # Maximum tokens for generation
TEMPERATURE = 0.2      # Creativity level (0-1)
```

## How It Works

### Pipeline Overview

1. **Data Loading**: Loads HumanEval dataset with 164 diverse coding examples
2. **Embedding Generation**: Creates 384-dimensional embeddings using sentence transformers
3. **Vector Store**: Builds FAISS index with L2-normalized vectors for cosine similarity
4. **Smart Retrieval**: Finds most semantically similar examples for any coding task
5. **Context-Aware Generation**: Uses LLM with retrieved examples as context
6. **Quality Assurance**: Validates generated code with automated testing

### Example Workflow

```python
# 1. Setup pipeline
pipeline = setup_rag_pipeline()

# 2. Generate code for Fibonacci sequence
task = "Write a function to calculate the fibonacci sequence"
result = generate_code_for_task(pipeline, task)

# 3. Get results
print(f"Generated code: {result['generated_code']}")
print(f"Similarity scores: {[ex['similarity'] for ex in result['retrieved_examples']]}")
```

## Example Output

### Input Task

```python
def calculate_median(numbers: List[float]) -> float:
    """Calculate the median of a list of numbers."""
```

### Generated Code

```python
from typing import List

def calculate_median(numbers: List[float]) -> float:
    """Calculate the median of a list of numbers."""
    numbers = sorted(numbers)
    n = len(numbers)
    if n % 2 == 1:
        return float(numbers[n // 2])
    else:
        return (numbers[n // 2 - 1] + numbers[n // 2]) / 2.0
```

### Retrieval Results

- **HumanEval/47** (0.8026 similarity): Direct median function example
- **HumanEval/4** (0.6628 similarity): Statistical function with similar structure
- **HumanEval/21** (0.5341 similarity): List processing with type hints

## Performance

### Retrieval Accuracy

- **High Similarity Scores**: Cosine similarity up to 0.80+
- **Relevant Context**: Always finds semantically related examples
- **Fast Search**: FAISS enables millisecond-level retrieval

## Error Handling

The system gracefully handles:

- Missing API keys (falls back to search-only mode)
- Network timeouts
- Invalid generated code syntax
- Empty retrieval results


## Acknowledgments

- OpenAI for the HumanEval dataset
- Facebook Research for FAISS
- SentenceTransformers for embedding models
- OpenRouter for LLM API access
