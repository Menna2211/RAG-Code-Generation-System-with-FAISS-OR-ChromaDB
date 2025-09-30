import faiss
from faiss_index import build_faiss_index , search_similar
from embeddings import load_embedding_model, create_embeddings


def print_generation_result(result):
    """Print the code generation results in a nice format."""
    print("\n" + "="*80)
    print(" TASK DESCRIPTION:")
    print("="*80)
    print(result["task_description"])
    
    print("\n" + "="*80)
    print(" GENERATED CODE:")
    print("="*80)
    print(result["generated_code"])
    
    print("\n" + "="*80)
    print(" RETRIEVED EXAMPLES:")
    print("="*80)
    for i, example in enumerate(result["retrieved_examples"], 1):
        print(f"\n{i}. {example['task_id']} (similarity: {example['similarity']:.4f})")
        print(f"   {example['prompt'][:150]}...")


def compare_similarity_methods(query, examples, embedding_model):
    """
    Compare L2 distance vs cosine similarity for educational purposes.
    """
    print("\n" + "="*80)
    print(" Comparing Similarity Methods")
    print("="*80)
    
    # Use first 10 examples for demo
    demo_examples = examples[:10]
    prompts = [ex['prompt'] for ex in demo_examples]
    embeddings = create_embeddings(embedding_model, prompts)
    
    # Method 1: L2 Distance
    print("\n1. L2 Distance (measures straight-line distance):")
    l2_index = faiss.IndexFlatL2(embeddings.shape[1])
    l2_index.add(embeddings)
    
    query_emb = embedding_model.encode([query]).astype('float32')
    l2_distances, l2_indices = l2_index.search(query_emb, 3)
    
    for i, (idx, dist) in enumerate(zip(l2_indices[0], l2_distances[0]), 1):
        print(f"   {i}. {demo_examples[idx]['task_id']} (distance: {dist:.2f})")
    
    # Method 2: Cosine Similarity (our method)
    print("\n2. Cosine Similarity (measures angle similarity):")
    cosine_index = build_faiss_index(embeddings)
    
    similarities, indices = search_similar(cosine_index, query_emb, 3)
    
    for i, (idx, sim) in enumerate(zip(indices[0], similarities[0]), 1):
        print(f"   {i}. {demo_examples[idx]['task_id']} (similarity: {sim:.4f})")
    
    print("\n Note: Higher cosine similarity = more similar")
    print("   Lower L2 distance = more similar")
    print("="*80)