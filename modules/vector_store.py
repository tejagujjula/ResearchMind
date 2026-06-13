import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document

def create_vector_store(chunks, embedding_model, persist_dir: str = "chroma_db", collection_name: str = "researchmind_collection") -> Chroma:
    """
    Clears the existing persist directory, extracts text and metadata from chunks,
    creates unique chunk IDs, and stores them in a persistent ChromaDB vector store.
    
    Args:
        chunks: List of chunk dictionaries containing 'text', 'source', 'page', and 'chunk_id'.
        embedding_model: The langchain embedding model instance.
        persist_dir: Directory where ChromaDB should persist data.
        collection_name: Name of the ChromaDB collection.
        
    Returns:
        Chroma: The instantiated langchain Chroma object.
    """
    try:
        # Clear existing chroma_db directory to avoid duplicate vectors during development
        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
            except Exception as delete_err:
                # If directory is locked or deletion fails, we log it and proceed
                pass
                
        # Ensure target directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        # Prepare LangChain Document objects and unique IDs
        documents = []
        ids = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk.get("source", "Unknown"),
                    "page": chunk.get("page", 0),
                    "chunk_id": chunk.get("chunk_id", 0)
                }
            )
            documents.append(doc)
            # Unique ID representation (e.g., chunk_1, chunk_2)
            ids.append(f"chunk_{chunk.get('chunk_id', 0)}")
            
        # Store documents in ChromaDB with fixed collection name
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_dir,
            collection_name=collection_name,
            ids=ids
        )
        
        return vector_store
        
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store in ChromaDB: {str(e)}")

def get_collection_stats(vector_store) -> dict:
    """
    Retrieves the total number of documents in the collection using count()
    and the collection name.
    
    Args:
        vector_store: The Chroma vector store object.
        
    Returns:
        dict: {'total_documents': int, 'collection_name': str}
    """
    try:
        collection = vector_store._collection
        count = collection.count()
        name = collection.name
        return {
            "total_documents": count,
            "collection_name": name
        }
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve vector store statistics: {str(e)}")
