import streamlit as st
import os
from langchain_chroma import Chroma
from modules.pdf_processor import save_uploaded_file, extract_text_from_pdf
from modules.text_splitter import chunk_documents
from modules.embeddings import get_embedding_model, generate_embeddings
from modules.vector_store import create_vector_store, get_collection_stats
from modules.retriever import retrieve_relevant_chunks
from modules.groq_llm import get_llm, generate_answer
from modules.memory_manager import get_memory, add_to_memory, get_chat_history, clear_memory
from modules.summarizer import generate_paper_summary
from modules.quiz_generator import generate_quiz
from modules.analytics_manager import (
    initialize_analytics,
    update_papers_uploaded,
    update_pages_processed,
    update_chunks_created,
    update_questions_asked,
    update_answers_generated,
    update_summaries_generated,
    update_quizzes_generated,
    update_comparisons_generated,
    update_query_rewrites_performed,
    update_retrieval_distances,
    update_conversation_turns,
    get_analytics_summary
)
from modules.comparison_manager import get_unique_sources, compare_papers
from modules.query_rewriter import rewrite_query
import plotly.express as px
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="ResearchMind",
    page_icon="📚",
    layout="wide"
)

# Initialize analytics manager
initialize_analytics()

# Initialize memory using Streamlit session state
if "memory" not in st.session_state:
    try:
        st.session_state.memory = get_memory()
    except Exception as e:
        st.error(f"Failed to initialize conversation memory: {str(e)}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "generated_summaries" not in st.session_state:
    st.session_state.generated_summaries = {
        "abstract": None,
        "methodology": None,
        "findings": None,
        "limitations": None
    }

if "latest_summary_type" not in st.session_state:
    st.session_state.latest_summary_type = None

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

if "comparison_metadata" not in st.session_state:
    st.session_state.comparison_metadata = {"papers": [], "dimension": ""}

# Load existing vector store on startup if present
persist_dir = "chroma_db"
if "vector_store" not in st.session_state:
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            emb_model = get_embedding_model()
            st.session_state.vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=emb_model,
                collection_name="researchmind_collection"
            )
        except Exception as e:
            pass

# --- Top Landing Section ---
st.title("📚 ResearchMind")
st.subheader("AI-Powered Research Paper Assistant")
st.write(
    "Welcome to ResearchMind! An advanced research assistant featuring semantic search, "
    "RAG pipeline question answering, targeted summarization, automated quiz generation, multi-PDF comparison, "
    "and usage analytics."
)

# Quick stats row
summary = get_analytics_summary()
q_col1, q_col2, q_col3, q_col4 = st.columns(4)
with q_col1:
    st.metric(label="Total Papers", value=summary["papers_uploaded"])
with q_col2:
    st.metric(label="Chunks Generated", value=summary["chunks_created"])
with q_col3:
    st.metric(label="Questions Asked", value=summary["questions_asked"])
with q_col4:
    st.metric(label="Active Memory Turns", value=summary["conversation_turns"])
st.divider()

# --- Sidebar Configuration & Controls ---
st.sidebar.title("⚙️ Configuration")

# 1. Groq API Configuration
# During deployment on Streamlit Community Cloud, configure the GROQ_API_KEY secret:
# GROQ_API_KEY = "your_groq_api_key"
# via: Streamlit Cloud -> App Settings -> Secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = None

st.session_state.groq_api_key = groq_api_key

# 2. Retrieval Configuration
st.sidebar.divider()
st.sidebar.subheader("Retrieval Settings")
query_optimization_enabled = st.sidebar.checkbox(
    "Enable Query Optimization",
    value=True,
    help="Rewrite queries internally to improve retrieval results."
)

# 3. Compact Status Indicators
st.sidebar.divider()
st.sidebar.subheader("System Status")

vs_status = "🟢 Ready" if "vector_store" in st.session_state and st.session_state.vector_store is not None else "🔴 Not Ready"
st.sidebar.write(f"**Vector Store:** {vs_status}")

groq_status = "🟢 Connected" if st.session_state.get("groq_api_key") else "🔴 Missing"
st.sidebar.write(f"**Groq API:** {groq_status}")

turns = len(st.session_state.get("chat_history", []))
mem_status = f"🟢 Active ({turns} turns)" if turns > 0 else "⚪ Empty"
st.sidebar.write(f"**Conversation Memory:** {mem_status}")

# 4. Action Controls
st.sidebar.divider()
st.sidebar.subheader("Session Controls")

if st.sidebar.button("🧹 Clear Conversation", help="Clear conversational history and reset memory context"):
    try:
        clear_memory(st.session_state.memory)
        st.session_state.chat_history = []
        st.sidebar.success("Conversation history cleared successfully.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to clear conversation: {str(e)}")

if st.sidebar.button("🔄 Reset Analytics", help="Reset all analytics counters for the current session"):
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
    st.sidebar.success("Analytics dashboard reset successfully.")
    st.rerun()


# --- Navigation Tabs ---
tab_docs, tab_qa, tab_sum, tab_quiz, tab_compare, tab_analytics, tab_arch, tab_dev = st.tabs([
    "📄 Document Processing",
    "🔍 Ask Questions",
    "📝 Summaries",
    "🧠 Quiz Mode",
    "📚 Multi-PDF Comparison",
    "📊 Analytics Dashboard",
    "🏗️ Architecture",
    "👨‍💻 About Developer"
])

# ==========================================
# Tab 1: Document Processing
# ==========================================
with tab_docs:
    st.subheader("Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="main_pdf_uploader"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} PDF(s) selected.")
        
        # Explicit processing trigger button
        process_clicked = st.button("🚀 Process Documents", key="process_docs_btn")
        
        if process_clicked:
            processed_count = 0
            total_pages = 0
            all_pages_data = []
            processing_errors = []
            
            # 1. Processing (Saving and extracting)
            with st.spinner("Processing uploaded PDFs (saving and extracting text)..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Save PDF locally
                        saved_path = save_uploaded_file(uploaded_file)
                        
                        # Extract text page by page
                        pages_data = extract_text_from_pdf(saved_path)
                        
                        processed_count += 1
                        total_pages += len(pages_data)
                        all_pages_data.extend(pages_data)
                        
                    except Exception as e:
                        processing_errors.append(f"Error processing '{uploaded_file.name}': {str(e)}")

            # Display results or errors
            if processing_errors:
                for err in processing_errors:
                    st.error(err)
                    
            if processed_count > 0:
                # 2. Chunking Documents
                chunks = []
                chunking_error = None
                with st.spinner("Splitting extracted text into semantic chunks..."):
                    try:
                        chunks = chunk_documents(all_pages_data)
                        st.session_state.chunks = chunks
                    except Exception as e:
                        chunking_error = f"Error during document chunking: {str(e)}"
                        
                if chunking_error:
                    st.error(chunking_error)
                else:
                    # 3. Generating Embeddings
                    vectors = []
                    embedding_error = None
                    if chunks:
                        with st.spinner("Generating embedding vectors using HuggingFace model (all-MiniLM-L6-v2)..."):
                            try:
                                _, vectors = generate_embeddings(chunks)
                            except Exception as e:
                                embedding_error = f"Error during embedding generation: {str(e)}"
                    
                    if embedding_error:
                        st.error(embedding_error)
                    elif chunks:
                        # 4. ChromaDB Storage
                        vector_store_instance = None
                        db_error = None
                        with st.spinner("Storing document chunks and embedding vectors in persistent ChromaDB..."):
                            try:
                                # Fetch the embedding model function
                                emb_model = get_embedding_model()
                                # Create vector store
                                vector_store_instance = create_vector_store(chunks, emb_model)
                                # Save to session state
                                st.session_state.vector_store = vector_store_instance
                            except Exception as e:
                                db_error = f"Error storing chunks in ChromaDB: {str(e)}"
                        
                        if db_error:
                            st.error(db_error)
                            if "processed_files_info" in st.session_state:
                                del st.session_state.processed_files_info
                            if "last_processed_filenames" in st.session_state:
                                del st.session_state.last_processed_filenames
                        elif vector_store_instance:
                            # Update analytics counters
                            update_papers_uploaded(processed_count)
                            update_pages_processed(total_pages)
                            update_chunks_created(len(chunks))
                            
                            # Gather collection statistics
                            db_stats = None
                            try:
                                db_stats = get_collection_stats(vector_store_instance)
                            except Exception as e:
                                db_stats = None
                            
                            # Calculate statistics
                            chunk_lengths = [len(c["text"]) for c in chunks]
                            avg_chunk_length = sum(chunk_lengths) / len(chunks) if chunks else 0
                            min_chunk_length = min(chunk_lengths) if chunks else 0
                            max_chunk_length = max(chunk_lengths) if chunks else 0
                            
                            total_embeddings = len(vectors)
                            dimension_size = len(vectors[0]) if total_embeddings > 0 else 0
                            
                            first_chunk = chunks[0] if chunks else None
                            first_chunk_info = None
                            if first_chunk:
                                first_chunk_info = {
                                    "source": first_chunk["source"],
                                    "page": first_chunk["page"],
                                    "chunk_id": first_chunk["chunk_id"],
                                    "text": first_chunk["text"]
                                }
                            
                            saved_locations = []
                            for uploaded_file in uploaded_files:
                                expected_path = os.path.join("data", "uploaded_pdfs", uploaded_file.name)
                                saved_locations.append({
                                    "name": uploaded_file.name,
                                    "path": expected_path,
                                    "exists": os.path.exists(expected_path)
                                })
                            
                            # Cache processing results in session state
                            st.session_state.processed_files_info = {
                                "processed_count": processed_count,
                                "total_pages": total_pages,
                                "chunks_count": len(chunks),
                                "avg_chunks": len(chunks) / total_pages if total_pages > 0 else 0,
                                "avg_chunk_length": avg_chunk_length,
                                "min_chunk_length": min_chunk_length,
                                "max_chunk_length": max_chunk_length,
                                "total_embeddings": total_embeddings,
                                "dimension_size": dimension_size,
                                "db_stats": db_stats,
                                "first_chunk_info": first_chunk_info,
                                "saved_locations": saved_locations
                            }
                            
                            # Cache matching filenames
                            st.session_state.last_processed_filenames = [f.name for f in uploaded_files]
                            
                            # Clear old session report data since new papers have been processed
                            st.session_state.generated_summaries = {
                                "abstract": None,
                                "methodology": None,
                                "findings": None,
                                "limitations": None
                            }
                            st.session_state.latest_summary_type = None
                            st.session_state.quiz_questions = []
                            st.session_state.comparison_result = None
                            st.session_state.comparison_metadata = {"papers": [], "dimension": ""}
                            
                            # Trigger rerun so landing/top metrics update and uploader remains populated
                            st.rerun()

        # Display cached metrics/statistics ONLY if uploader matches the last processed files
        current_filenames = [f.name for f in uploaded_files]
        last_processed_filenames = st.session_state.get("last_processed_filenames", [])
        
        if current_filenames == last_processed_filenames and "processed_files_info" in st.session_state:
            info = st.session_state.processed_files_info
            
            st.success("Successfully processed PDFs, generated embeddings, and stored chunks in persistent ChromaDB!")
            
            # Display core metrics
            st.write("### Document Processing Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="PDFs Processed", value=info["processed_count"])
            with col2:
                st.metric(label="Total Pages Extracted", value=info["total_pages"])
            with col3:
                st.metric(label="Total Chunks Generated", value=info["chunks_count"])
            with col4:
                st.metric(label="Avg Chunks / Page", value=f"{info['avg_chunks']:.2f}")
            
            # Display Chunk statistics
            st.write("### Chunk Statistics")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric(label="Average Chunk Length", value=f"{info['avg_chunk_length']:.1f} chars")
            with stats_col2:
                st.metric(label="Minimum Chunk Length", value=f"{info['min_chunk_length']} chars")
            with stats_col3:
                st.metric(label="Maximum Chunk Length", value=f"{info['max_chunk_length']} chars")
            
            # Display Embedding metrics
            st.write("### Embedding Vector Metrics")
            emb_col1, emb_col2 = st.columns(2)
            with emb_col1:
                st.metric(label="Total Embeddings Generated", value=info["total_embeddings"])
            with emb_col2:
                st.metric(label="Embedding Dimension Size", value=info["dimension_size"])
            
            # Display ChromaDB metrics
            if info["db_stats"]:
                st.write("### ChromaDB Vector Store Status")
                db_col1, db_col2, db_col3 = st.columns(3)
                with db_col1:
                    st.metric(label="Total Vectors Stored", value=info["db_stats"]["total_documents"])
                with db_col2:
                    st.metric(label="ChromaDB Collection Name", value=info["db_stats"]["collection_name"])
                with db_col3:
                    st.metric(label="Database Persistence Dir", value="chroma_db/")
            
            # Display preview of the first chunk
            if info["first_chunk_info"]:
                first_chunk = info["first_chunk_info"]
                st.write("### First Chunk Preview (First 500 Characters):")
                st.info(
                    f"**Source File:** `{first_chunk['source']}` | "
                    f"**Page:** {first_chunk['page']} | "
                    f"**Chunk ID:** {first_chunk['chunk_id']}"
                )
                
                chunk_text_preview = first_chunk["text"]
                if len(chunk_text_preview) > 500:
                    chunk_text_preview = chunk_text_preview[:500] + "\n... [truncated] ..."
                st.code(chunk_text_preview, language="text")
                
            # List physical paths of saved PDFs for reference
            st.write("### Saved Document Locations:")
            for loc in info["saved_locations"]:
                if loc["exists"]:
                    st.caption(f"💾 Saved to: `{loc['path']}`")
    else:
        if "vector_store" in st.session_state and st.session_state.vector_store is not None:
            st.info("💡 Research papers are already stored in ChromaDB database. Feel free to search, compare, or run analytics using the other tabs.")
        else:
            st.info("ℹ️ Please upload research papers (PDF files only) to get started.")


# ==========================================
# Tab 2: Ask Questions (RAG QA)
# ==========================================
with tab_qa:
    st.subheader("🔍 Query & Retrieve Documents")
    vector_store = st.session_state.get("vector_store")
    if vector_store:
        query = st.text_input("Ask a question about your research papers", placeholder="Enter your search query or question...", key="qa_query_input")
        retrieve_button = st.button("Retrieve Relevant Information", key="qa_retrieve_btn")
        
        # Distance threshold configuration (Lower distance indicates higher similarity.)
        DISTANCE_THRESHOLD = 1.2
        
        if retrieve_button:
            if not query.strip():
                st.warning("Please enter a query to search.")
            else:
                # Increment questions asked in analytics
                update_questions_asked()
                
                # Rewrite query internally if enabled
                search_query = query
                api_key = st.session_state.get("groq_api_key")
                
                if query_optimization_enabled and api_key:
                    with st.spinner("Optimizing search query with Groq (Llama)..."):
                        try:
                            llm_rewrite = get_llm(api_key)
                            chat_history_str = get_chat_history(st.session_state.memory)
                            search_query = rewrite_query(llm_rewrite, query, chat_history_str)
                            
                            # Increment queries optimized if rewrite actually altered the query
                            if search_query != query:
                                update_query_rewrites_performed()
                        except Exception:
                            search_query = query
                
                with st.spinner("Searching ChromaDB for relevant information..."):
                    try:
                        # Retrieve the top 5 chunks using the optimized query and apply threshold
                        retrieval_data = retrieve_relevant_chunks(vector_store, search_query, k=5, distance_threshold=DISTANCE_THRESHOLD)
                        retrieved_chunks = retrieval_data["results"]
                        raw_count = retrieval_data["total_before_filtering"]
                        remaining_count = len(retrieved_chunks)
                        
                        # Update analytics for retrieval distances
                        scores = [c["score"] for c in retrieved_chunks]
                        if scores:
                            update_retrieval_distances(scores)
                        
                        # Display Query Optimization if enabled
                        if query_optimization_enabled and api_key:
                            with st.expander("🔍 Query Optimization", expanded=False):
                                st.markdown(f"**Original Query:** `{query}`")
                                st.markdown(f"**Optimized Retrieval Query:** `{search_query}`")
                                
                        st.write("### Retrieval Statistics")
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        with m_col1:
                            st.metric(label="Chunks Retrieved (Raw)", value=raw_count)
                        with m_col2:
                            st.metric(label="Chunks Remaining (Filtered)", value=remaining_count)
                        with m_col3:
                            st.metric(label="Applied Distance Threshold", value=DISTANCE_THRESHOLD)
                        with m_col4:
                            if remaining_count > 0:
                                avg_dist = sum([c["score"] for c in retrieved_chunks]) / remaining_count
                                st.metric(label="Avg Similarity Distance", value=f"{avg_dist:.4f}")
                            else:
                                st.metric(label="Avg Similarity Distance", value="N/A")
                                
                        # Answer Generation Logic
                        if remaining_count > 0:
                            st.write("### AI Answer Generation")
                            
                            if api_key:
                                with st.spinner("Generating answer with Groq (Llama 3)..."):
                                    try:
                                        llm = get_llm(api_key)
                                        # Retrieve formatted chat history from memory
                                        chat_history_str = get_chat_history(st.session_state.memory)
                                        
                                        # Generate grounded answer incorporating history
                                        answer = generate_answer(llm, query, retrieved_chunks, chat_history_str)
                                        # Increment answers generated in analytics
                                        update_answers_generated()
                                        
                                        # Display AI Generated Answer in a highlighted container
                                        st.success("🤖 AI Generated Answer")
                                        with st.container(border=True):
                                            st.markdown(answer)
                                            
                                        # Save to conversation memory if the answer is not a fallback response
                                        fallback_msg = "I could not find sufficient information in the uploaded research papers."
                                        if answer.strip() != fallback_msg:
                                            add_to_memory(st.session_state.memory, query, answer)
                                            from datetime import datetime
                                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            st.session_state.chat_history.append({
                                                "question": query,
                                                "answer": answer,
                                                "timestamp": timestamp
                                            })
                                            # Restrict Streamlit chat history list to last 5 interactions
                                            st.session_state.chat_history = st.session_state.chat_history[-5:]
                                            # Update conversation turns in analytics
                                            update_conversation_turns()
                                            st.rerun() # Refresh layout to update landing turns metrics
                                            
                                        # Display unique sources used
                                        unique_sources = sorted(list({(chunk["source"], chunk["page"]) for chunk in retrieved_chunks}))
                                        st.write("**Sources Used:**")
                                        for src, pg in unique_sources:
                                            st.write(f"- 📄 {src} (Page {pg})")
                                            
                                    except Exception as llm_err:
                                        st.error(f"Failed to generate answer: {str(llm_err)}")
                            else:
                                st.warning("Groq API key is not configured. Please configure Streamlit Secrets.")
                                
                        if retrieved_chunks:
                            st.write("### Retrieved Semantic Chunks")
                            # Display each chunk in an expander
                            for i, chunk in enumerate(retrieved_chunks, start=1):
                                expander_header = (
                                    f"📄 Chunk {i} | Source: {chunk['source']} | "
                                    f"Page: {chunk['page']} | Chunk ID: {chunk['chunk_id']} | "
                                    f"Similarity Distance: {chunk['score']:.4f}"
                                )
                                with st.expander(expander_header):
                                    st.write(chunk["text"])
                        else:
                            st.info("No sufficiently relevant information was found in the uploaded research papers.")
                    except Exception as e:
                        st.error(f"Retrieval failed: {str(e)}")
        
        # Display previous interactions using Streamlit expanders
        if st.session_state.get("chat_history"):
            st.divider()
            st.subheader("💬 Conversation History")
            
            from modules.report_generator import generate_chat_report
            chat_pdf = generate_chat_report(st.session_state.chat_history)
            st.download_button(
                label="📥 Download Conversation Report",
                data=chat_pdf,
                file_name="conversation_history.pdf",
                mime="application/pdf",
                key="download_chat_btn"
            )
            
            for i, chat in enumerate(st.session_state.chat_history, start=1):
                with st.expander(f"Interaction {i}: {chat['question'][:80]}..."):
                    st.write(f"**User:** {chat['question']}")
                    st.write(f"**Assistant:** {chat['answer']}")
    else:
        st.info("Please upload research papers in the 'Document Processing' tab first.")


# ==========================================
# Tab 3: Summaries
# ==========================================
with tab_sum:
    st.subheader("📝 Research Paper Summaries")
    st.write("Generate targeted summaries using the first 10 chunks of the research paper generated in this session.")
    
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        btn_abstract = st.button("Generate Abstract Summary", key="sum_abstract_btn")
    with sum_col2:
        btn_methodology = st.button("Generate Methodology Summary", key="sum_method_btn")
    with sum_col3:
        btn_findings = st.button("Generate Key Findings", key="sum_findings_btn")
    with sum_col4:
        btn_limitations = st.button("Generate Limitations", key="sum_limitations_btn")
    
    selected_type = None
    header_title = None
    emoji_title = None
    
    if btn_abstract:
        selected_type = "abstract"
        header_title = "Abstract Summary"
        emoji_title = "📄"
    elif btn_methodology:
        selected_type = "methodology"
        header_title = "Methodology Summary"
        emoji_title = "🔬"
    elif btn_findings:
        selected_type = "findings"
        header_title = "Key Findings"
        emoji_title = "📊"
    elif btn_limitations:
        selected_type = "limitations"
        header_title = "Limitations"
        emoji_title = "⚠️"
    
    if selected_type:
        # 1. Verify vector store and session chunks exist
        if "vector_store" not in st.session_state or "chunks" not in st.session_state:
            st.warning("Please upload and process a research paper first to generate summaries.")
        # 2. Verify API key exists
        elif not st.session_state.get("groq_api_key"):
            st.warning("Groq API key is not configured. Please configure Streamlit Secrets.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    summary_chunks = st.session_state.chunks
                    api_key = st.session_state.groq_api_key
                    llm = get_llm(api_key)
                    
                    # Generate summary
                    summary_text = generate_paper_summary(llm, summary_chunks, selected_type)
                    update_summaries_generated()
                    
                    # Cache in session state
                    st.session_state.generated_summaries[selected_type] = summary_text
                    st.session_state.latest_summary_type = selected_type
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate summary: {str(e)}")

    # Persistent summaries rendering
    latest_type = st.session_state.get("latest_summary_type")
    if latest_type and st.session_state.generated_summaries.get(latest_type):
        summary_text = st.session_state.generated_summaries[latest_type]
        
        # Mapping to titles/emojis
        titles = {
            "abstract": ("Abstract Summary", "📄"),
            "methodology": ("Methodology Summary", "🔬"),
            "findings": ("Key Findings", "📊"),
            "limitations": ("Limitations", "⚠️")
        }
        h_title, e_title = titles[latest_type]
        
        st.subheader(f"{e_title} {h_title}")
        with st.container(border=True):
            st.markdown(summary_text)

    # Download button for generated summaries
    if any(st.session_state.generated_summaries.values()):
        st.write("")
        from modules.report_generator import generate_summary_report
        summary_pdf = generate_summary_report(st.session_state.generated_summaries)
        st.download_button(
            label="📥 Download Summary Report",
            data=summary_pdf,
            file_name="research_summaries.pdf",
            mime="application/pdf",
            key="download_summary_btn"
        )


# ==========================================
# Tab 4: Quiz Mode
# ==========================================
with tab_quiz:
    st.subheader("🧠 Research Quiz Mode")
    st.write("Generate a multiple-choice quiz based on the first 10 chunks of the research paper to test your understanding.")
    
    generate_quiz_btn = st.button("Generate Quiz", key="quiz_generate_btn")
    
    if generate_quiz_btn:
        # 1. Verify vector store and session chunks exist
        if "vector_store" not in st.session_state or "chunks" not in st.session_state:
            st.warning("Please upload and process a research paper first to generate a quiz.")
        # 2. Verify API key exists
        elif not st.session_state.get("groq_api_key"):
            st.warning("Groq API key is not configured. Please configure Streamlit Secrets.")
        else:
            with st.spinner("Generating quiz..."):
                try:
                    summary_chunks = st.session_state.chunks
                    api_key = st.session_state.groq_api_key
                    llm = get_llm(api_key)
                    
                    # Generate quiz list (default 5 questions)
                    quiz_questions = generate_quiz(llm, summary_chunks, num_questions=5)
                    
                    if quiz_questions:
                        st.session_state.quiz_questions = quiz_questions
                        update_quizzes_generated()
                        st.rerun()
                    else:
                        st.error("No questions were generated. Please try again.")
                except Exception as e:
                    st.error(f"Failed to generate quiz: {str(e)}")

    if st.session_state.get("quiz_questions"):
        # Download button for quiz
        from modules.report_generator import generate_quiz_report
        quiz_pdf = generate_quiz_report(st.session_state.quiz_questions)
        st.download_button(
            label="📥 Download Quiz Report",
            data=quiz_pdf,
            file_name="research_quiz.pdf",
            mime="application/pdf",
            key="download_quiz_btn"
        )
        
        st.write("### Multiple Choice Quiz")
        # Display each MCQ inside styled containers
        for i, mcq in enumerate(st.session_state.quiz_questions, start=1):
            with st.container(border=True):
                st.markdown(f"**Question {i}:** {mcq.get('question', '')}")
                
                # Options display
                options = mcq.get("options", {})
                for letter, text in options.items():
                    st.write(f"**{letter}.** {text}")
                    
                st.write("")
                
                # Structured Correct answer display inside an expander
                correct_ans = mcq.get("correct_answer", {})
                ans_option = correct_ans.get("option", "")
                ans_text = correct_ans.get("text", "")
                
                with st.expander("🔑 Reveal Correct Answer & Explanation"):
                    st.success(f"**Correct Answer:** {ans_option} - {ans_text}")
                    st.info(f"**Explanation:** {mcq.get('explanation', '')}")


# ==========================================
# Tab 5: Multi-PDF Comparison
# ==========================================
with tab_compare:
    st.subheader("📚 Multi-PDF Comparison")
    st.write("Compare uploaded research papers across methodologies, key findings, limitations, contributions, or an overall comparison.")
    
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        available_papers = get_unique_sources(st.session_state.vector_store)
        if available_papers:
            selected_papers = st.multiselect(
                "Select papers to compare",
                options=available_papers,
                help="Select 2 or more papers for side-by-side analysis",
                key="compare_papers_select"
            )
            
            comparison_dimension = st.selectbox(
                "Select comparison dimension",
                options=["Overall Comparison", "Methodology", "Key Findings", "Limitations", "Contributions"],
                key="compare_dim_select"
            )
            
            compare_button = st.button("Compare Papers", key="compare_action_btn")
            
            if compare_button:
                if not selected_papers or len(selected_papers) < 2:
                    st.warning("Please select at least 2 research papers to perform a comparison.")
                elif not st.session_state.get("groq_api_key"):
                    st.warning("Groq API key is not configured. Please configure Streamlit Secrets.")
                else:
                    with st.spinner("Performing comparative analysis..."):
                        try:
                            api_key = st.session_state.groq_api_key
                            llm = get_llm(api_key)
                            
                            comparison_result = compare_papers(
                                llm,
                                st.session_state.vector_store,
                                selected_papers,
                                comparison_dimension
                            )
                            
                            # Cache in session state
                            st.session_state.comparison_result = comparison_result
                            st.session_state.comparison_metadata = {
                                "papers": selected_papers,
                                "dimension": comparison_dimension
                            }
                            
                            # Update analytics comparisons count
                            update_comparisons_generated()
                            st.rerun()
                        except Exception as comp_err:
                            st.error(f"Failed to generate comparison: {str(comp_err)}")
                            
            if st.session_state.get("comparison_result"):
                st.write("")
                from modules.report_generator import generate_comparison_report
                meta = st.session_state.comparison_metadata
                comp_pdf = generate_comparison_report(
                    st.session_state.comparison_result,
                    papers=meta.get("papers"),
                    dimension=meta.get("dimension")
                )
                st.download_button(
                    label="📥 Download Comparison Report",
                    data=comp_pdf,
                    file_name="pdf_comparison_report.pdf",
                    mime="application/pdf",
                    key="download_compare_btn"
                )
                
                st.write(f"### Comparative Report: {meta.get('dimension')}")
                with st.container(border=True):
                    st.markdown(st.session_state.comparison_result)
        else:
            st.info("No research papers were found in the database. Please upload papers to enable comparison.")
    else:
        st.info("Please upload research papers first to enable comparisons.")


# ==========================================
# Tab 6: Analytics Dashboard
# ==========================================
with tab_analytics:
    st.subheader("📊 Analytics Dashboard")
    st.write("Track and visualize your usage metrics for the current session.")
    
    summary = get_analytics_summary()
    
    # Row 1: Document metrics
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Papers Uploaded", value=summary["papers_uploaded"])
    with m_col2:
        st.metric(label="Pages Processed", value=summary["pages_processed"])
    with m_col3:
        st.metric(label="Chunks Created", value=summary["chunks_created"])
    
    # Row 2: Interaction metrics
    m_col4, m_col5, m_col6 = st.columns(3)
    with m_col4:
        st.metric(label="Questions Asked", value=summary["questions_asked"])
    with m_col5:
        st.metric(label="AI Answers Generated", value=summary["answers_generated"])
    with m_col6:
        st.metric(label="Conversation Turns", value=summary["conversation_turns"])
    
    # Row 3: Feature metrics
    m_col7, m_col8, m_col9, m_col10 = st.columns(4)
    with m_col7:
        st.metric(label="Summaries Generated", value=summary["summaries_generated"])
    with m_col8:
        st.metric(label="Quizzes Generated", value=summary["quizzes_generated"])
    with m_col9:
        st.metric(label="Comparisons Generated", value=summary["comparisons_generated"])
    with m_col10:
        st.metric(label="Queries Optimized", value=summary["query_rewrites_performed"])
    
    # Row 4: Advanced metrics
    m_col11, = st.columns(1)
    with m_col11:
        avg_dist = summary["average_retrieval_distance"]
        val_str = f"{avg_dist:.4f}" if avg_dist > 0 else "N/A"
        st.metric(label="Avg Retrieval Distance", value=val_str)
    
    # Visualizations wrapped in an expander
    show_pie = (
        summary["questions_asked"] > 0 or 
        summary["answers_generated"] > 0 or 
        summary["summaries_generated"] > 0 or 
        summary["quizzes_generated"] > 0 or
        summary["comparisons_generated"] > 0 or
        summary["query_rewrites_performed"] > 0
    )
    show_hist = len(st.session_state.analytics.get("retrieval_distances", [])) > 0
    
    if show_pie or show_hist:
        st.divider()
        with st.expander("📊 View Session Visualizations", expanded=True):
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                if show_pie:
                    # Data preparation
                    df_pie = pd.DataFrame({
                        "Activity": ["Questions Asked", "Answers Generated", "Summaries Generated", "Quizzes Generated", "Comparisons Generated", "Queries Optimized"],
                        "Count": [
                            summary["questions_asked"],
                            summary["answers_generated"],
                            summary["summaries_generated"],
                            summary["quizzes_generated"],
                            summary["comparisons_generated"],
                            summary["query_rewrites_performed"]
                        ]
                    })
                    # Filter rows where Count > 0
                    df_pie = df_pie[df_pie["Count"] > 0]
                    if not df_pie.empty:
                        fig_pie = px.pie(
                            df_pie, 
                            values="Count", 
                            names="Activity", 
                            title="User Activity Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_pie.update_layout(margin=dict(t=50, b=10, l=10, r=10))
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
            with chart_col2:
                if show_hist:
                    distances = st.session_state.analytics["retrieval_distances"]
                    df_hist = pd.DataFrame({"Distance": distances})
                    fig_hist = px.histogram(
                        df_hist, 
                        x="Distance", 
                        title="Retrieval Similarity Distance Distribution",
                        labels={"Distance": "Similarity Distance"},
                        color_discrete_sequence=["#1f77b4"]
                    )
                    fig_hist.update_layout(margin=dict(t=50, b=10, l=10, r=10), yaxis_title="Count")
                    st.plotly_chart(fig_hist, use_container_width=True)

    # Check if there is any data to export for the complete report
    has_summaries = any(st.session_state.get("generated_summaries", {}).values())
    has_chat = len(st.session_state.get("chat_history", [])) > 0
    has_quiz = len(st.session_state.get("quiz_questions", [])) > 0
    has_comparison = st.session_state.get("comparison_result") is not None
    
    if has_summaries or has_chat or has_quiz or has_comparison:
        st.divider()
        st.subheader("📥 Export Session Report")
        st.write("Download a comprehensive PDF report containing all summaries, conversation logs, comparative analyses, and quizzes generated during this active session.")
        
        from modules.report_generator import generate_complete_report
        complete_pdf = generate_complete_report(
            chat_history=st.session_state.get("chat_history", []),
            summary_data=st.session_state.get("generated_summaries", {}),
            quiz_data=st.session_state.get("quiz_questions", []),
            comparison_text=st.session_state.get("comparison_result"),
            comparison_meta=st.session_state.get("comparison_metadata")
        )
        
        st.download_button(
            label="📥 Download Complete ResearchMind Report",
            data=complete_pdf,
            file_name="researchmind_complete_session_report.pdf",
            mime="application/pdf",
            key="download_complete_btn"
        )




# ==========================================
# Tab 7: Architecture
# ==========================================
with tab_arch:
    st.subheader("🏗️ ResearchMind Architecture")
    
    st.markdown("### Project Overview")
    st.write(
        "ResearchMind is a modular, AI-powered research assistant that enables users to upload scientific papers, "
        "perform semantic searches, generate answers strictly grounded in paper contexts, create targeted section-based "
        "summaries, generate quizzes for comprehension assessment, perform side-by-side paper comparisons, "
        "and export findings to custom PDFs."
    )
    
    st.markdown("### System Workflow Pipeline")
    # Bordered workflow diagram with emojis/icons
    with st.container(border=True):
        st.markdown(
            "```\n"
            "   [ 📂 PDF Upload ] &rarr; Saves to data/uploaded_pdfs/\n"
            "          ↓\n"
            "   [ 📝 Text Extraction ] &rarr; Page-by-page parsing via pdfplumber\n"
            "          ↓\n"
            "   [ ✂️ Document Chunking ] &rarr; RecursiveCharacterTextSplitter (size=500, overlap=100)\n"
            "          ↓\n"
            "   [ 🧠 Embedding Gen ] &rarr; Local sentence-transformers (all-MiniLM-L6-v2)\n"
            "          ↓\n"
            "   [ 🗄️ ChromaDB Storage ] &rarr; Persistent vector index in chroma_db/\n"
            "          ↓\n"
            "   [ 🔄 Query Rewrite ] &rarr; Academic query expansion & pronoun resolution (Groq)\n"
            "          ↓\n"
            "   [ 🔎 Semantic Retrieval ] &rarr; Top k=5/6 chunks with Cosine Distance threshold <= 1.2\n"
            "          ↓\n"
            "   [ 🤖 Context Grounding ] &rarr; Prompt-engineering to isolate context & prevent LLM hallucination\n"
            "          ↓\n"
            "   [ 💬 Answer Generation ] &rarr; Groq Cloud (Llama 3 70B) with exact citations\n"
            "          ↓\n"
            "   [ 🧠 Memory Buffer ] &rarr; Conversation history buffer tracking the last 5 turns\n"
            "          ↓\n"
            "   [ 📊 Analytics Dashboard ] &rarr; Real-time Plotly charts & session metrics tracking\n"
            "          ↓\n"
            "   [ 📥 PDF Report Export ] &rarr; Custom ReportLab Platypus report compiler\n"
            "```",
            unsafe_allow_html=True
        )
        
    st.markdown("### Technology Stack & Component Architecture")
    # Expanded Tech table with Purpose column
    tech_data = {
        "Component": ["Frontend", "LLM API", "Embeddings", "Vector Database", "Orchestration", "PDF Parsing", "Report Generation"],
        "Technology": ["Streamlit", "Groq (Llama 3 70B)", "HuggingFace (all-MiniLM-L6-v2)", "ChromaDB", "LangChain", "pdfplumber", "ReportLab"],
        "Purpose": [
            "Provides a responsive, tabbed user interface, interactive inputs, and analytics metrics display.",
            "Powers sub-second, context-grounded response generation, summarization, quiz generation, and query expansion.",
            "Computes dense vector representations of parsed document chunks locally without external network calls.",
            "Indexes, searches, and persistently stores document chunk embeddings locally.",
            "Integrates prompt templates, semantic retriever pipelines, and conversational buffer memory context.",
            "Parses character coordinates and extracts raw text from uploaded PDFs page-by-page.",
            "Programmatically compiles session metrics, summaries, comparisons, and Q&A history into standard PDFs."
        ]
    }
    st.table(tech_data)
    
    st.markdown("### ResearchMind at a Glance")
    # Stats summary grid inside bordered container
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Core Specifications:**")
            st.markdown("- **Embeddings Model:** `all-MiniLM-L6-v2` (Local)")
            st.markdown("- **LLM Model:** `llama3-70b-8192` (Groq Cloud)")
            st.markdown("- **Vector Store Persistence:** `chroma_db/` (Active)")
        with col2:
            st.markdown("**Processing Configs:**")
            st.markdown("- **Chunk Size:** 500 characters")
            st.markdown("- **Chunk Overlap:** 100 characters")
            st.markdown("- **Distance Threshold:** <= 1.2 (Cosine Distance)")
        with col3:
            st.markdown("**Memory & Session Settings:**")
            st.markdown("- **Buffer Max Size:** Last 5 Q&A turns")
            st.markdown("- **Chroma Collection:** `researchmind_collection`")
            st.markdown("- **PDF Margins:** 54pt (0.75 inch)")

    st.markdown("### Features Implemented")
    st.markdown("""
- **Local PDF Saving**: Created a `data/uploaded_pdfs` directory and saved all uploaded files while preserving their original names.
- **Robust Text Extraction**: Integrated `pdfplumber` to extract text page-by-page.
- **Automatic Document Chunking**: Implemented `RecursiveCharacterTextSplitter` with high precision parameters (`chunk_size=500`, `chunk_overlap=100`).
- **Local Embedding Generation**: Integrated Hugging Face's `all-MiniLM-L6-v2` model via `langchain_huggingface` to generate vector representations locally.
- **ChromaDB Persistence**: Utilized `langchain_chroma` integration to clean and store all chunks and embeddings in a persistent directory (`chroma_db/`) under the collection name `researchmind_collection`. Deletes old database directories prior to rebuilding.
- **Optimized Semantic Retrieval**: Configured query retrieval to fetch `k=5` chunks and apply a `DISTANCE_THRESHOLD = 1.2` distance filter, calculating pre- and post-filtering counts and average similarity distance.
- **AI Answer Generation**: Integrated Groq Llama 3 (`llama3-70b-8192`) to answer user questions using only the retrieved paper context. Contains strict hallucination constraints, API key validation (`gsk_` check), and details the exact citations/sources used.
- **Conversational Memory Integration**: Integrated LangChain `ConversationBufferMemory` to maintain the context of the last 5 successful interactions. Prompt templates were enhanced to leverage conversation history specifically for resolving references (pronouns, follow-ups) while keeping answers strictly grounded in retrieved document facts. Users can view history through expanders and clear the memory state at any time.
- **Automatic Research Paper Summarization**: Added capability to generate structured summaries for Abstract, Methodology, Key Findings, and Limitations. The application utilizes the first 10 document chunks generated in the active session and formats custom tasks for Groq ChatGroq LLM, presenting results in bordered layout containers.
- **Research Quiz Mode**: Implemented automated generation of 5 multiple-choice questions from the first 10 document chunks. Leverages regex-based JSON extraction and a structured `correct_answer` schema. Quizzes are rendered using Streamlit expanders with distinct display areas for the question, options A-D, correct option text, and reasoning explanation.
- **Interactive Analytics Dashboard**: Configured session-based metrics tracking across all activities (documents, Q&A, summaries, and quizzes). Provides a modular analytics manager, a Reset dashboard option, and Plotly visualization charts including an Activity Distribution Pie Chart and a Retrieval Quality Histogram (rendering distributions and averages without hardcoded thresholds).
- **Multi-PDF Comparison**: Supports filtered semantic search comparison across multiple documents. Side-by-side analysis covers Objectives, Methodologies, Key Findings, Strengths, Limitations, and preferable scenarios.
- **Semantic Query Optimization**: Integrates a dynamic rewriting layer using Groq Llama for pronoun reference resolution and academic expansion of vague queries, with toggle activation and analytics tracking.
- **UI/UX Polishing**: Reorganized the layout into clean functional tabs, restructured the sidebar configuration and controls, introduced compact emoji-based status indicators, formatted outputs in clean bordered containers, and added quick landing metrics and a professional footer.
""")


# ==========================================
# Tab 8: About Developer
# ==========================================
with tab_dev:
    st.subheader("👨‍💻 About the Developer")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True):
            st.markdown("### Developer Details")
            st.markdown("**Name:** Teja Reddy Gujjula")
            st.markdown("**Email:** [tejareddy_gujjula18@gmail.com](mailto:tejareddy_gujjula18@gmail.com)")
            st.markdown("**LinkedIn:** [linkedin.com/in/teja-reddy](https://www.linkedin.com/in/teja-reddy-g/)")
            st.markdown("**GitHub:** [github.com/tejagujjula](https://github.com/tejagujjula)")
    with col2:
        with st.container(border=True):
            st.markdown("### Project Description")
            st.write(
                "ResearchMind was developed as an end-to-end AI-powered research assistant to simplify research paper analysis "
                "through Retrieval-Augmented Generation (RAG), semantic search, summarization, multi-document comparison, "
                "and intelligent report generation."
            )
            st.write(
                "The project is structured following clean coding principles, with decoupled backend modules for processing, "
                "vector storage, LLM orchestration, and report compiling, making it completely modular, "
                "portfolio-ready, and extensible."
            )

# Professional footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "<strong>ResearchMind v1.0</strong> – Transforming Research through AI-powered Insights.<br>"
    "<em>Features: Semantic Search • RAG Pipeline • Multi-PDF Analysis • Query Optimization • Quiz Generation • PDF Exports</em><br>"
    "Built using Streamlit • ChromaDB • LangChain • Groq LLM • ReportLab"
    "</div>",
    unsafe_allow_html=True
)