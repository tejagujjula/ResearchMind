from typing import List, Dict, Any
from langchain_core.documents import Document

def get_unique_sources(vector_store) -> List[str]:
    """
    Retrieves unique source names (filenames) stored in the ChromaDB vector store.
    """
    try:
        if not vector_store:
            return []
        collection = vector_store._collection
        db_data = collection.get(include=["metadatas"])
        metadatas = db_data.get("metadatas", [])
        if not metadatas:
            return []
        sources = {meta.get("source") for meta in metadatas if meta and meta.get("source")}
        return sorted(list(sources))
    except Exception as e:
        return []

def compare_papers(llm, vector_store, selected_papers: List[str], dimension: str) -> str:
    """
    Performs filtered retrieval (k=6) for each selected paper and invokes Groq
    to generate a structured comparative analysis.
    """
    if not vector_store:
        raise ValueError("Vector store is not initialized.")
    if not selected_papers or len(selected_papers) < 2:
        raise ValueError("At least 2 papers must be selected for comparison.")
        
    normalized_dim = dimension.lower().strip()
    
    # Dimension-specific search queries
    queries = {
        "methodology": "methodology approach architecture algorithms systems experimental design",
        "key findings": "key findings results achievements experiments performance metrics",
        "limitations": "limitations weaknesses future work challenges issues bottlenecks",
        "contributions": "contributions novelty impact objectives benefits overall contribution",
        "overall comparison": "overview objective methodology key findings limitations strengths weaknesses summary"
    }
    
    search_query = queries.get(normalized_dim, f"details about {normalized_dim}")
    
    try:
        # Retrieve context for each selected paper
        paper_contexts = []
        for paper in selected_papers:
            # Query vector store with source filter
            results = vector_store.similarity_search(
                search_query,
                k=6,
                filter={"source": paper}
            )
            
            # Combine retrieved chunks for this paper
            chunks_text = []
            for i, doc in enumerate(results, start=1):
                chunks_text.append(f"[Excerpt {i}]:\n{doc.page_content}")
                
            combined_excerpts = "\n\n".join(chunks_text) if chunks_text else "No relevant information retrieved."
            
            paper_contexts.append(
                f"=== Paper: {paper} ===\n"
                f"Retrieved Context:\n{combined_excerpts}"
            )
            
        full_context_block = "\n\n===========================================\n\n".join(paper_contexts)
        
        # Construct the system instruction and prompt template
        system_instruction = (
            "You are ResearchMind, an AI research assistant. Your task is to compare the selected research papers "
            "based ONLY on the provided paper contexts. Follow these strict rules:\n"
            "1. Rely ONLY on the clear facts mentioned in the context. Do NOT use outside knowledge.\n"
            "2. Do NOT hallucinate, extrapolate, or assume facts not directly stated.\n"
            "3. If details for a specific paper are missing or context is insufficient under the comparison dimension, state that clearly."
        )
        
        if normalized_dim == "overall comparison":
            prompt = (
                f"{system_instruction}\n\n"
                f"Context Blocks:\n{full_context_block}\n\n"
                f"Task: Generate a comprehensive Overall Comparison of the papers based ONLY on the context blocks. Your analysis must compare them across:\n"
                f"- Objectives\n"
                f"- Methodologies\n"
                f"- Key Findings\n"
                f"- Strengths\n"
                f"- Limitations\n\n"
                f"Conclude with a clear recommendation on when/why each paper's approach or methodology may be preferable over the other.\n\n"
                f"Structure your response clearly with Markdown headers, comparison tables, or clear comparative bullet points.\n\n"
                f"Comparison Report:"
            )
        else:
            prompt = (
                f"{system_instruction}\n\n"
                f"Context Blocks:\n{full_context_block}\n\n"
                f"Task: Generate a structured comparative analysis focusing ONLY on the dimension: '{dimension}'.\n"
                f"Explain similarities, differences, and unique characteristics of each paper regarding this dimension.\n\n"
                f"Structure your comparison using clear markdown comparison tables or comparative bullet points.\n\n"
                f"Comparison Report:"
            )
            
        # Invoke Groq model
        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        raise RuntimeError(f"Error during comparative analysis: {str(e)}")
