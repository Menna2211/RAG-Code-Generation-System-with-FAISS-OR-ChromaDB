import numpy as np
import faiss


def build_faiss_index(embeddings, doc_ids=None):
    """
    Build a FAISS index with normalized vectors for cosine similarity.
    
    This uses Inner Product on normalized vectors, which gives us cosine similarity.
    It's better than L2 distance for text similarity.
    """
    print("Building FAISS index for cosine similarity...")
    
    dimension = embeddings.shape[1]
    
    # Normalize the vectors (important for cosine similarity)
    normalized_embeddings = embeddings.copy()
    faiss.normalize_L2(normalized_embeddings)
    print(f"✓ Normalized {len(normalized_embeddings)} vectors")
    
    # Create document IDs if not provided
    if doc_ids is None:
        doc_ids = np.arange(len(embeddings)).astype('int64')
    
    # Create index with Inner Product (cosine similarity)
    base_index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(normalized_embeddings, doc_ids)
    
    print(f"✓ FAISS index built with {index.ntotal} vectors")
    return index


def search_similar(index, query_embedding, k=3):
    """Search for similar vectors in the index."""
    # Normalize the query vector
    normalized_query = query_embedding.copy()
    faiss.normalize_L2(normalized_query)
    
    # Search (returns similarity scores)
    similarities, indices = index.search(normalized_query, k)
    
    return similarities[0], indices[0]


def save_faiss_index(index, filepath):
    """Save the FAISS index to disk."""
    faiss.write_index(index, filepath)
    print(f"✓ Index saved to {filepath}")


def load_faiss_index(filepath):
    """Load a FAISS index from disk."""
    index = faiss.read_index(filepath)
    print(f"✓ Index loaded from {filepath}")
    return index