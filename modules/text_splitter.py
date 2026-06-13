from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Splits the text of input documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: A list of dicts where each dict has:
            - "source": PDF filename (str)
            - "page": page number (int)
            - "text": extracted page text (str)
            
    Returns:
        A list of chunk dicts:
        {
            "source": PDF filename,
            "page": page number,
            "chunk_id": unique sequential integer (1-indexed),
            "text": chunk text
        }
    """
    try:
        # Initialize the splitter with the required settings
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        chunked_results = []
        chunk_counter = 1
        
        for doc in documents:
            source = doc.get("source", "Unknown")
            page = doc.get("page", 0)
            text = doc.get("text", "")
            
            # Skip pages/documents without text
            if not text or not text.strip():
                continue
                
            # Split the page text
            chunks = splitter.split_text(text)
            
            for chunk in chunks:
                # Skip empty chunks
                if chunk and chunk.strip():
                    chunked_results.append({
                        "source": source,
                        "page": page,
                        "chunk_id": chunk_counter,
                        "text": chunk.strip()
                    })
                    chunk_counter += 1
                    
        return chunked_results
        
    except Exception as e:
        # Wrap any unexpected split errors in a descriptive RuntimeError
        raise RuntimeError(f"Error occurred during text splitting: {str(e)}")
