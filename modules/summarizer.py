from typing import List, Dict, Any
from langchain_groq import ChatGroq

def generate_paper_summary(llm: ChatGroq, chunks: List[Dict[str, Any]], summary_type: str) -> str:
    """
    Generates a targeted research paper summary based on the first 8-10 chunks.
    
    Args:
        llm: Initialized ChatGroq model instance.
        chunks: List of document chunks (each containing 'text', etc.).
        summary_type: One of 'abstract', 'methodology', 'findings', 'limitations'.
        
    Returns:
        str: Generated summary text.
    """
    if not chunks:
        return "No paper chunks available to generate a summary."
        
    # Map summary type to its corresponding prompt
    prompts = {
        "abstract": "Provide a concise summary of the research paper, including its objective and overall contribution.",
        "methodology": "Summarize the methodology used in this research paper. Explain the approach taken by the authors.",
        "findings": "Summarize the key findings and important results reported in this research paper.",
        "limitations": "Identify and summarize the limitations or weaknesses discussed in this research paper. If none are mentioned, state that explicitly."
    }
    
    normalized_type = summary_type.lower().strip()
    if normalized_type not in prompts:
        raise ValueError(f"Invalid summary_type: '{summary_type}'. Must be one of 'abstract', 'methodology', 'findings', 'limitations'.")
        
    task_prompt = prompts[normalized_type]
    
    try:
        # Combine the first 8-10 chunks into a context block
        selected_chunks = chunks[:10]
        context_parts = []
        for i, chunk in enumerate(selected_chunks, start=1):
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            context_parts.append(f"Chunk {i} Content:\n{text}")
            
        context_block = "\n\n---\n\n".join(context_parts)
        
        # Structure the instructions for the Groq LLM
        system_instruction = (
            "You are ResearchMind, an AI research assistant. Your task is to answer the summarization request "
            "using ONLY the provided research paper context. Follow these strict rules:\n"
            "1. Rely ONLY on the clear facts mentioned in the context. Do NOT use any outside knowledge.\n"
            "2. Do NOT hallucinate, extrapolate, or assume any facts not directly stated.\n"
            "3. Answer the request precisely and professionally."
        )
        
        prompt = (
            f"{system_instruction}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Request:\n{task_prompt}\n\n"
            f"Summary:"
        )
        
        # Invoke Groq model
        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        raise RuntimeError(f"Error during summarization generation: {str(e)}")
