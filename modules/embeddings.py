from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Tuple, Any

# Module-level cache to prevent reloading the model on every function invocation
_embedding_model_cache = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFaceEmbeddings model: sentence-transformers/all-MiniLM-L6-v2.
    Caches the model instance for subsequent calls.
    
    Returns:
        HuggingFaceEmbeddings: The initialized embedding model.
    """
    global _embedding_model_cache
    if _embedding_model_cache is None:
        try:
            _embedding_model_cache = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFaceEmbeddings model: {str(e)}")
    return _embedding_model_cache

def generate_embeddings(chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """
    Generates embedding vectors for a list of document chunks.
    
    Args:
        chunks: A list of chunk dictionaries:
            [
                {
                    "source": "paper.pdf",
                    "page": 1,
                    "chunk_id": 1,
                    "text": "Chunk text"
                }
            ]
            
    Returns:
        Tuple:
            - List[Dict[str, Any]]: The original chunks input.
            - List[List[float]]: List of embedding vectors corresponding to the chunks.
            
    Raises:
        RuntimeError: If embedding generation fails.
    """
    if not chunks:
        return [], []
        
    try:
        model = get_embedding_model()
        
        # Extract text from each chunk
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Generate embeddings for all texts
        embedding_vectors = model.embed_documents(texts)
        
        return chunks, embedding_vectors
        
    except Exception as e:
        raise RuntimeError(f"Error during embedding generation: {str(e)}")
