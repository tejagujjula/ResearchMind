import streamlit as st
from typing import Dict, Any, List

def initialize_analytics() -> None:
    """
    Safely initializes analytics tracking structure in Streamlit session state if not present.
    """
    try:
        if "analytics" not in st.session_state:
            st.session_state.analytics = {
                "papers_uploaded": 0,
                "pages_processed": 0,
                "chunks_created": 0,
                "questions_asked": 0,
                "answers_generated": 0,
                "summaries_generated": 0,
                "quizzes_generated": 0,
                "comparisons_generated": 0,
                "query_rewrites_performed": 0,
                "retrieval_distances": [],
                "conversation_turns": 0
            }
    except Exception as e:
        st.error(f"Error initializing analytics manager: {str(e)}")

def update_papers_uploaded(count: int) -> None:
    """
    Increments the papers uploaded counter.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["papers_uploaded"] += count
    except Exception as e:
        pass

def update_pages_processed(count: int) -> None:
    """
    Increments the total PDF pages processed counter.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["pages_processed"] += count
    except Exception as e:
        pass

def update_chunks_created(count: int) -> None:
    """
    Increments the total document chunks generated counter.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["chunks_created"] += count
    except Exception as e:
        pass

def update_questions_asked() -> None:
    """
    Increments the questions asked counter.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["questions_asked"] += 1
    except Exception as e:
        pass

def update_answers_generated() -> None:
    """
    Increments the AI answers generated counter.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["answers_generated"] += 1
    except Exception as e:
        pass

def update_summaries_generated() -> None:
    """
    Increments the summaries generated counter.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["summaries_generated"] += 1
    except Exception as e:
        pass

def update_quizzes_generated() -> None:
    """
    Increments the quizzes generated counter.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["quizzes_generated"] += 1
    except Exception as e:
        pass

def update_comparisons_generated() -> None:
    """
    Increments the comparisons generated counter.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["comparisons_generated"] += 1
    except Exception as e:
        pass

def update_query_rewrites_performed() -> None:
    """
    Increments the query rewrites performed counter.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["query_rewrites_performed"] += 1
    except Exception as e:
        pass

def update_retrieval_distances(distances: List[float]) -> None:
    """
    Appends new retrieval similarity distances to the tracking list.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["retrieval_distances"].extend(distances)
    except Exception as e:
        pass

def update_conversation_turns() -> None:
    """
    Increments the active conversation turns counter in memory.
    """
    try:
        initialize_analytics()
        st.session_state.analytics["conversation_turns"] += 1
    except Exception as e:
        pass

def get_analytics_summary() -> Dict[str, Any]:
    """
    Calculates summary metrics such as averages and returns the structured summary.
    """
    try:
        initialize_analytics()
        data = st.session_state.analytics
        distances = data.get("retrieval_distances", [])
        
        # Calculate average retrieval similarity distance
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        
        return {
            "papers_uploaded": data["papers_uploaded"],
            "pages_processed": data["pages_processed"],
            "chunks_created": data["chunks_created"],
            "questions_asked": data["questions_asked"],
            "answers_generated": data["answers_generated"],
            "summaries_generated": data["summaries_generated"],
            "quizzes_generated": data["quizzes_generated"],
            "comparisons_generated": data.get("comparisons_generated", 0),
            "query_rewrites_performed": data.get("query_rewrites_performed", 0),
            "average_retrieval_distance": avg_distance,
            "conversation_turns": data["conversation_turns"]
        }
    except Exception as e:
        # Return empty/default metrics dict on error
        return {
            "papers_uploaded": 0,
            "pages_processed": 0,
            "chunks_created": 0,
            "questions_asked": 0,
            "answers_generated": 0,
            "summaries_generated": 0,
            "quizzes_generated": 0,
            "comparisons_generated": 0,
            "query_rewrites_performed": 0,
            "average_retrieval_distance": 0.0,
            "conversation_turns": 0
        }
