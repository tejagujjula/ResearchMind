# 🧠 ResearchMind

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-AI%20App-red.svg)
![RAG](https://img.shields.io/badge/RAG-Enabled-orange.svg)
![Groq](https://img.shields.io/badge/Groq-LLM-green.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

> **AI-Powered Research Paper Assistant built using RAG, Groq LLM, ChromaDB, and Streamlit.**

ResearchMind is an intelligent research assistant that helps users interact with research papers through semantic search, conversational question answering, targeted summarization, automated quiz generation, multi-paper comparison, and analytics dashboards.

---

## 🎯 Why ResearchMind?

Research papers are often lengthy and difficult to analyze efficiently. ResearchMind leverages **Retrieval-Augmented Generation (RAG)** and **Large Language Models** to help researchers, students, and professionals extract insights, generate summaries, compare papers, and test understanding through quizzes—all within an intuitive interface.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/tejagujjula/ResearchMind.git
cd ResearchMind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Groq API Key

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Alternatively, you can enter the API key directly through the application's sidebar.

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser at:

```text
http://localhost:8501
```

---

## 🚀 Features

### 📄 Document Processing

* Upload multiple research papers (PDFs)
* Extract text page-by-page using PyMuPDF
* Automatic document chunking
* Local embedding generation using Sentence Transformers
* ChromaDB vector storage for semantic retrieval

### 🔍 Intelligent Question Answering

* Retrieval-Augmented Generation (RAG) pipeline
* Conversational memory support
* Context-aware question answering
* Citation-grounded responses

### 🧠 Query Optimization

* Dynamic query rewriting using Groq LLM
* Pronoun resolution for follow-up questions
* Semantic query expansion for improved retrieval quality

### 📝 Research Summarization

Generate targeted summaries for:

* Abstract
* Methodology
* Key Findings
* Limitations

### 🎯 Quiz Generation

* Automatic MCQ generation from research papers
* Correct answer identification
* Explanation for each question

### 📚 Multi-PDF Comparison

Compare multiple papers across dimensions such as:

* Objectives
* Methodologies
* Findings
* Strengths
* Limitations

### 📊 Analytics Dashboard

Track usage statistics including:

* Documents processed
* Chunks generated
* Questions asked
* Query optimizations performed
* Activity visualizations

### 📄 PDF Report Exports

Download reports for:

* Conversation History
* Research Summaries
* Generated Quizzes
* Comparison Reports

---

## 🏗️ System Architecture

```text
PDF Upload
     ↓
Text Extraction
     ↓
Document Chunking
     ↓
Embedding Generation
     ↓
ChromaDB Vector Storage
     ↓
Query Rewriting (Optional)
     ↓
Semantic Retrieval
     ↓
Groq LLM Generation
     ↓
Answer / Summary / Quiz / Comparison
```

---

## 🛠️ Tech Stack

| Category             | Technologies                               |
| -------------------- | ------------------------------------------ |
| Frontend             | Streamlit                                  |
| LLM                  | Groq (Llama 3)                             |
| Framework            | LangChain                                  |
| Vector Database      | ChromaDB                                   |
| Embeddings           | Sentence Transformers (`all-MiniLM-L6-v2`) |
| PDF Processing       | PyMuPDF                                    |
| Visualization        | Plotly                                     |
| Report Generation    | ReportLab                                  |
| Programming Language | Python                                     |

---

## 📂 Project Structure

```text
ResearchMind/
│
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── analytics_manager.py
│   ├── comparison_manager.py
│   ├── embeddings.py
│   ├── groq_llm.py
│   ├── memory_manager.py
│   ├── pdf_processor.py
│   ├── query_rewriter.py
│   ├── quiz_generator.py
│   ├── report_generator.py
│   ├── retriever.py
│   ├── summarizer.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── chroma_db/
└── data/
```

---

## ⚙️ Installation

### Prerequisites

* Python 3.8 or higher
* Groq API Key (available from https://console.groq.com)

### Clone Repository

```bash
git clone https://github.com/tejagujjula/ResearchMind.git
cd ResearchMind
```

### Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

Create a `.streamlit/secrets.toml` file in the project root:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Alternatively, you can provide the API key directly within the application sidebar.

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The application launches automatically in your default browser:

```text
http://localhost:8501
```

---

## 📖 How to Use

1. Upload one or more research papers (PDF format).
2. Click **Process Documents** to generate embeddings and build the vector database.
3. Navigate to **Ask Questions** to perform semantic queries.
4. Generate **Summaries** for abstracts, methodologies, findings, and limitations.
5. Use **Quiz Mode** to test your understanding with automatically generated MCQs.
6. Compare multiple papers using **Multi-PDF Comparison**.
7. Export conversations, summaries, quizzes, and comparison reports as PDFs.
8. Monitor usage through the **Analytics Dashboard**.

---

## 📸 Application Preview

Screenshots showcasing Document Processing, Question Answering, Summarization, Quiz Generation, Multi-PDF Comparison, and Analytics Dashboard will be added soon.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience with:

* Retrieval-Augmented Generation (RAG)
* Large Language Model Integration
* Semantic Search Systems
* Vector Databases
* Conversational AI
* Prompt Engineering
* Streamlit Application Development
* Git & GitHub Workflows

---

## 🤝 Contributing

Contributions are welcome!

If you have suggestions, feature requests, or improvements:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

Please feel free to open an issue for bugs or enhancement ideas.

---

## 🌟 Future Improvements

### Short-Term Goals

* User Authentication
* Improved citation highlighting
* Enhanced UI customization
* Support for additional export formats

### Long-Term Goals

* Research Paper Recommendation Engine
* Citation Graph Visualization
* Cloud Vector Database Support
* Research Collaboration Features
* Multi-user Workspace Support

---

## 👨‍💻 Developer

**Teja Reddy**

📧 Email: [tejareddy_gujjula18@gmail.com](mailto:tejareddy_gujjula18@gmail.com)

🔗 LinkedIn: https://www.linkedin.com/in/teja-reddy-g/

💻 GitHub: https://github.com/tejagujjula

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!
