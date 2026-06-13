from typing import Dict, Any

def retrieve_relevant_chunks(vector_store, query: str, k: int = 5, distance_threshold: float = 1.2) -> Dict[str, Any]:
    """
    Searches ChromaDB using similarity search with score, filters by a distance threshold,
    and returns both the filtered results and retrieval metadata.
    
    Args:
        vector_store: The instantiated langchain Chroma object.
        query: The user's search query string.
        k: The number of top relevant chunks to retrieve before filtering.
        distance_threshold: The maximum distance score allowed (lower distance indicates higher similarity).
        
    Returns:
        A dictionary containing:
        {
            "results": List[Dict[str, Any]] (filtered chunks),
            "total_before_filtering": int (count of chunks retrieved before filtering),
            "distance_threshold": float (the threshold used)
        }
    """
    # Lower distance indicates higher similarity.
    if not vector_store:
        raise ValueError("Vector store is not initialized. Please upload documents first.")
        
    if not query or not query.strip():
        return {"results": [], "total_before_filtering": 0, "distance_threshold": distance_threshold}
        
    try:
        # Perform similarity search with score. Returns a list of tuples: (Document, score)
        results = vector_store.similarity_search_with_score(query, k=k)
        raw_count = len(results)
        
        filtered_chunks = []
        for doc, score in results:
            # Lower distance indicates higher similarity.
            if score <= distance_threshold:
                metadata = doc.metadata or {}
                filtered_chunks.append({
                    "text": doc.page_content,
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", 0),
                    "chunk_id": metadata.get("chunk_id", 0),
                    "score": float(score)  # Ensure score is standard float
                })
                
        return {
            "results": filtered_chunks,
            "total_before_filtering": raw_count,
            "distance_threshold": distance_threshold
        }
        
    except Exception as e:
        raise RuntimeError(f"Error occurred during similarity search: {str(e)}")
