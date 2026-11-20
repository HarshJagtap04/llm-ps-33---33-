# Textbook Doubt Solver (RAG Mini-Project)

Students get stuck while self-studying. This application answers doubts strictly using an uploaded textbook chapter PDF via Retrieval-Augmented Generation (RAG), providing a simple explanation, a verbatim quoted source line, and an **"Explain Simpler"** follow-up button.

## Tech Stack
- **Frontend:** Streamlit
- **LLM:** Google Gemini 2.5 Flash (`google-genai`)
- **Embeddings & Vector Store:** `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS
- **PDF Parser:** `pypdf`

## How to Run Locally
1. Clone the repository and navigate into the folder:
   ```bash
   git clone <your-repo-url>
   cd textbook-doubt-solver