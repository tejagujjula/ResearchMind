try:
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:
    from langchain.memory import ConversationBufferMemory
from typing import Any

def get_memory() -> ConversationBufferMemory:
    """
    Initialize and return a ConversationBufferMemory instance.
    
    Returns:
        ConversationBufferMemory: The initialized conversation memory helper.
    """
    try:
        return ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ConversationBufferMemory: {str(e)}")

def add_to_memory(memory: ConversationBufferMemory, question: str, answer: str) -> None:
    """
    Save the user question and AI answer into memory.
    Maintains only the last 5 interactions (10 messages).
    
    Args:
        memory: The ConversationBufferMemory instance.
        question: User's query string.
        answer: Generated AI answer.
    """
    try:
        memory.save_context({"input": question}, {"output": answer})
        # Restrict memory to the last 5 interactions (each has user/input and assistant/output message)
        if len(memory.chat_memory.messages) > 10:
            memory.chat_memory.messages = memory.chat_memory.messages[-10:]
    except Exception as e:
        raise RuntimeError(f"Failed to save context to memory: {str(e)}")

def get_chat_history(memory: ConversationBufferMemory) -> str:
    """
    Return formatted chat history as a string.
    
    Args:
        memory: The ConversationBufferMemory instance.
        
    Returns:
        str: Clean formatted chat string.
    """
    try:
        history_str = ""
        messages = memory.chat_memory.messages
        for msg in messages:
            role = "User" if msg.type == "human" else "Assistant"
            history_str += f"{role}: {msg.content}\n"
        return history_str.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to format chat history: {str(e)}")

def clear_memory(memory: ConversationBufferMemory) -> None:
    """
    Clear all stored conversations from memory.
    
    Args:
        memory: The ConversationBufferMemory instance.
    """
    try:
        memory.clear()
    except Exception as e:
        raise RuntimeError(f"Failed to clear memory: {str(e)}")

