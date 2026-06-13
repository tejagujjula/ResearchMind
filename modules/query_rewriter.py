import re
from typing import Optional
from langchain_groq import ChatGroq

def rewrite_query(llm: ChatGroq, query: str, chat_history: Optional[str] = None) -> str:
    """
    Intelligently rewrites user questions into optimized semantic search queries.
    Only rewrites if the query is short (<= 8 words) or contains pronouns.
    
    Args:
        llm: Initialized ChatGroq model.
        query: User's original query.
        chat_history: Formatted previous conversation history.
        
    Returns:
        str: Rewritten query, or original query if condition is not met or processing fails.
    """
    if not query or not query.strip():
        return query
        
    # Check if query is short (<= 8 words) or contains pronouns
    pronoun_set = {"it", "they", "this", "these", "them", "their", "its", "those", "he", "she", "him", "her", "his"}
    words = re.findall(r'\b\w+\b', query.lower())
    contains_pronoun = any(w in pronoun_set for w in words)
    word_count = len(query.split())
    
    # Skip rewriting if the query is long (> 8 words) and has no pronouns
    if word_count > 8 and not contains_pronoun:
        return query
        
    try:
        # Dynamically set temperature to 0.1 to minimize creativity during rewriting
        if hasattr(llm, "temperature"):
            llm.temperature = 0.1
            
        system_instruction = (
            "You are an expert query optimization assistant for Retrieval-Augmented Generation systems.\n"
            "Your task is to rewrite user questions into detailed semantic search queries that maximize retrieval effectiveness.\n\n"
            "Rules:\n"
            "1. Preserve the original intent exactly.\n"
            "2. Resolve pronouns using chat history only when necessary.\n"
            "3. Expand vague questions into precise research-oriented search queries.\n"
            "4. Do not answer the question.\n"
            "5. Return ONLY the rewritten query text.\n"
            "6. If the original query is already specific, return it unchanged."
        )
        
        history_text = chat_history if chat_history and chat_history.strip() else "No previous conversation."
        
        prompt = (
            f"{system_instruction}\n\n"
            f"Previous Conversation Context:\n{history_text}\n\n"
            f"Original Question:\n{query}\n\n"
            f"Rewritten Search Query:"
        )
        
        # Invoke Groq LLM
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        
        # In case the model returns empty or wraps it, clean it
        if rewritten:
            # Clean markdown formatting if any
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            return rewritten.strip()
            
        return query
        
    except Exception as e:
        # Fallback to the original query if anything fails
        return query
