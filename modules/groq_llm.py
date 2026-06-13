from langchain_groq import ChatGroq
from typing import List, Dict, Any

def get_llm(api_key: str) -> ChatGroq:
    """
    Initializes and returns the ChatGroq model with model llama3-70b-8192.
    
    Args:
        api_key: The Groq API key string.
        
    Returns:
        ChatGroq: The initialized LLM instance.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Groq API Key is missing or empty.")
        
    try:
        return ChatGroq(
            api_key=api_key,
            model="openai/gpt-oss-120b",
            temperature=0.2
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatGroq: {str(e)}")

def generate_answer(llm, query: str, retrieved_chunks: List[Dict[str, Any]], chat_history: str = None) -> str:
    """
    Generates a concise and accurate answer for a query based on retrieved chunks using Groq.
    Enforces strict context-only guidelines, uses conversation history for reference resolution only,
    and handles insufficient information fallbacks.
    
    Args:
        llm: The initialized ChatGroq LLM instance.
        query: The user's query string.
        retrieved_chunks: A list of retrieved chunk dictionaries.
        chat_history: Formatted chat history string.
        
    Returns:
        str: The generated answer string.
    """
    fallback_message = "I could not find sufficient information in the uploaded research papers."
    
    if not retrieved_chunks:
        return fallback_message
        
    try:
        # Construct context from retrieved chunks
        context_parts = []
        for chunk in retrieved_chunks:
            context_part = (
                f"Source: {chunk.get('source', 'Unknown')}\n"
                f"Page: {chunk.get('page', 0)}\n\n"
                f"Content:\n{chunk.get('text', '')}"
            )
            context_parts.append(context_part)
            
        context = "\n\n---\n\n".join(context_parts)
        
        # Format chat history representation
        formatted_history = chat_history if chat_history and chat_history.strip() else "No previous conversation."
        
        # Strengthened prompt template enforcing strict facts, context-only rules, and reference resolution limit
        prompt_text = (
            "You are ResearchMind, an AI research assistant. Your task is to answer the user's question "
            "using ONLY the provided Retrieved Context. Follow these strict rules:\n"
            "1. Rely ONLY on the clear facts mentioned in the Retrieved Context. Do NOT use any outside knowledge.\n"
            "2. Do NOT hallucinate, extrapolate, or assume any facts not directly stated.\n"
            "3. If the answer cannot be found in the Retrieved Context, or if the Retrieved Context is insufficient, you must reply "
            "with the exact phrase: 'I could not find sufficient information in the uploaded research papers.' and nothing else.\n"
            "4. Use the Previous Conversation history ONLY to resolve references (such as pronouns like 'it', 'they', 'this', or follow-up questions) in the Current Question. "
            "Do NOT use facts or information from the Previous Conversation to answer the Current Question; all facts must come strictly from the Retrieved Context.\n\n"
            "Provide a concise and accurate answer.\n\n"
            "Previous Conversation:\n{chat_history}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Current Question:\n{query}\n\n"
            "Answer:"
        )
        
        # Invoke Groq LLM
        prompt = prompt_text.format(chat_history=formatted_history, context=context, query=query)
        response = llm.invoke(prompt)
        
        return response.content.strip()
        
    except Exception as e:
        raise RuntimeError(f"Error during LLM answer generation: {str(e)}")

