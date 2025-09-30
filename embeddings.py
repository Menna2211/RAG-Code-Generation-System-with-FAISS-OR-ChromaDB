import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the sentence transformer model."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("✓ Model loaded successfully")
    return model


def create_embeddings(model, texts):
    """Create embeddings for a list of texts."""
    print(f"Creating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings_array = np.array(embeddings).astype('float32')
    
    print(f"✓ Created embeddings with shape: {embeddings_array.shape}")
    return embeddings_array


def embed_single_text(model, text):
    """Create embedding for a single piece of text."""
    return model.encode([text]).astype('float32')