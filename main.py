import os
from dotenv import load_dotenv

from config import *
from data_loader import load_humaneval_dataset, get_prompts_from_examples
from embeddings import load_embedding_model, create_embeddings, embed_single_text
from faiss_index import build_faiss_index, search_similar, save_faiss_index, load_faiss_index
from code_generator import setup_code_generator, generate_code
from utils import print_generation_result, compare_similarity_methods


def setup_rag_pipeline(api_key=None, embedding_model_name=EMBEDDING_MODEL):
    """
    Set up the complete RAG pipeline.
    Returns a dictionary with all the components.
    """
    print(" Setting up RAG Code Generator...")
    print("This might take a minute to load the models and data.\n")
    
    # Load environment variables
    load_dotenv()
    final_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    
    if not final_api_key:
        print(" Warning: No API key found. Code generation will not work.")
        print(" Please set OPENROUTER_API_KEY environment variable.")
    
    # Step 1: Load dataset
    examples = load_humaneval_dataset()
    prompts = get_prompts_from_examples(examples)
    
    # Step 2: Load embedding model and create embeddings
    embedding_model = load_embedding_model(embedding_model_name)
    embeddings = create_embeddings(embedding_model, prompts)
    
    # Step 3: Build FAISS index
    faiss_index = build_faiss_index(embeddings)
    
    # Step 4: Set up code generator client
    code_client = setup_code_generator(final_api_key) if final_api_key else None
    
    print("\n RAG pipeline is ready to use!")
    
    return {
        'examples': examples,
        'embedding_model': embedding_model,
        'faiss_index': faiss_index,
        'code_client': code_client,
        'api_key': final_api_key
    }


def generate_code_for_task(pipeline, task_description, num_examples=DEFAULT_NUM_EXAMPLES, 
                          model=GENERATION_MODEL):
    """
    Generate code for a task using the RAG pipeline.
    """
    print(f"\n Searching for {num_examples} similar examples...")
    
    # Find similar examples
    query_embedding = embed_single_text(pipeline['embedding_model'], task_description)
    similarities, indices = search_similar(pipeline['faiss_index'], query_embedding, num_examples)
    
    retrieved_examples = [pipeline['examples'][idx] for idx in indices]
    
    # Show what we found
    print(f" Found {len(retrieved_examples)} similar examples:")
    for i, (example, similarity) in enumerate(zip(retrieved_examples, similarities), 1):
        print(f"   {i}. {example['task_id']} (score: {similarity:.4f})")
    
    # Generate code if we have an API client
    if pipeline['code_client']:
        generated_code = generate_code(
            pipeline['code_client'],
            task_description,
            retrieved_examples,
            model=model
        )
    else:
        generated_code = "# API key required for code generation"
        print(" Skipping code generation - no API key available")
    
    # Return results
    return {
        "task_description": task_description,
        "generated_code": generated_code,
        "retrieved_examples": [
            {
                "task_id": ex['task_id'],
                'prompt': ex['prompt'],
                'canonical_solution': ex['canonical_solution'],
                'similarity': float(sim)
            }
            for ex, sim in zip(retrieved_examples, similarities)
        ]
    }


def save_pipeline_index(pipeline, filepath=FAISS_INDEX_PATH):
    """Save the FAISS index from the pipeline."""
    save_faiss_index(pipeline['faiss_index'], filepath)


def load_pipeline_with_index(filepath=FAISS_INDEX_PATH, api_key=None, embedding_model_name=EMBEDDING_MODEL):
    """Load a pipeline with a pre-built index."""
    load_dotenv()
    final_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    
    # Load the saved index
    faiss_index = load_faiss_index(filepath)
    
    # We still need examples and embedding model
    examples = load_humaneval_dataset()
    embedding_model = load_embedding_model(embedding_model_name)
    code_client = setup_code_generator(final_api_key) if final_api_key else None
    
    print("✅ Pipeline ready with loaded index!")
    
    return {
        'examples': examples,
        'embedding_model': embedding_model,
        'faiss_index': faiss_index,
        'code_client': code_client,
        'api_key': final_api_key
    }


# Simple example usage
if __name__ == "__main__":
    # Set up the complete pipeline
    pipeline = setup_rag_pipeline()
    
    # Example task - find the median of a list
    task = """
def calculate_median(numbers: List[float]) -> float:
    \"\"\" Calculate the median of a list of numbers.
    >>> calculate_median([3, 1, 2, 4, 5])
    3.0
    >>> calculate_median([1, 2, 3, 4])
    2.5
    \"\"\"
    """
    
    # Generate code
    result = generate_code_for_task(pipeline, task)
    
    # Show the results
    print_generation_result(result)
    
    # Save the index for later use
    save_pipeline_index(pipeline)
    
    print("\n Done! You can use this to generate code for any Python task.")

    # Test the pipeline with more tasks
    test_cases = [
        "Write a function to reverse a string without using built-in reverse",
        "Create a function that finds all prime numbers up to n",
        "Write a function to calculate the fibonacci sequence",
        "Create a function that checks if a number is a perfect square"
    ]

    for task in test_cases:
        result = generate_code_for_task(pipeline, task)
        print_generation_result(result)
        print("-" * 40)