import json
import re
from typing import List, Dict, Any
from langchain_groq import ChatGroq

def generate_quiz(llm: ChatGroq, chunks: List[Dict[str, Any]], num_questions: int = 5) -> List[Dict[str, Any]]:
    """
    Generates exactly num_questions multiple-choice questions based on the first 10 chunks.
    
    Args:
        llm: Initialized ChatGroq model instance.
        chunks: List of document chunks (each containing 'text', etc.).
        num_questions: Number of questions to generate (default 5).
        
    Returns:
        List[Dict[str, Any]]: List of question dictionaries.
    """
    if not chunks:
        raise ValueError("No paper chunks available to generate a quiz.")
        
    try:
        # Combine the first 10 chunks into a context block
        selected_chunks = chunks[:10]
        context_parts = []
        for i, chunk in enumerate(selected_chunks, start=1):
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            context_parts.append(f"Chunk {i} Content:\n{text}")
            
        context_block = "\n\n---\n\n".join(context_parts)
        
        # System guidelines
        system_instruction = (
            "You are ResearchMind, an AI research assistant. Your task is to generate a multiple-choice quiz "
            "using ONLY the provided research paper context. Follow these strict rules:\n"
            "1. Rely ONLY on the clear facts mentioned in the context. Do NOT use any outside knowledge.\n"
            "2. Do NOT hallucinate, extrapolate, or assume any facts not directly stated.\n"
            "3. Questions should test understanding of important concepts from the paper.\n"
            "4. Ensure options and correct answers are unambiguous.\n"
            "5. You must output exactly the requested number of questions.\n"
            "6. Output your response as a valid JSON object matching the JSON Schema below. Do NOT wrap it in markdown code blocks, do NOT add extra text or conversational filler, return ONLY the raw JSON."
        )
        
        json_schema = (
            "{\n"
            "  \"quiz\": [\n"
            "    {\n"
            "      \"question\": \"Question text here\",\n"
            "      \"options\": {\n"
            "        \"A\": \"Option A text\",\n"
            "        \"B\": \"Option B text\",\n"
            "        \"C\": \"Option C text\",\n"
            "        \"D\": \"Option D text\"\n"
            "      },\n"
            "      \"correct_answer\": {\n"
            "        \"option\": \"A\",\n"
            "        \"text\": \"Option A text\"\n"
            "      },\n"
            "      \"explanation\": \"Explanation text here\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        
        prompt = (
            f"{system_instruction}\n\n"
            f"JSON Schema:\n{json_schema}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Generate exactly {num_questions} multiple-choice questions based on the Context above.\n\n"
            f"Quiz JSON:"
        )
        
        # Invoke Groq model
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Robust regex-based extraction to extract the outer JSON block
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            quiz_json = match.group()
            quiz_data = json.loads(quiz_json)
            
            # Extract list of questions from "quiz" key
            if "quiz" in quiz_data and isinstance(quiz_data["quiz"], list):
                return quiz_data["quiz"]
            else:
                raise ValueError("JSON does not contain a 'quiz' key with a list of questions.")
        else:
            raise ValueError(f"No valid JSON structure found in the LLM response: {response_text}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to generate quiz: {str(e)}")
