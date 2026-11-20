import os
import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
st.set_page_config(page_title="Textbook Doubt Solver", page_icon="📚", layout="centered")
st.title("📚 Textbook Doubt Solver")
st.caption("Upload a textbook chapter PDF and ask doubts. Answers are strictly grounded in your chapter.")

if not api_key:
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()

client = Groq(api_key=api_key)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def extract_and_chunk_pdf(uploaded_file, chunk_size=500, overlap=100):
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            full_text += f"\n[Page {page_num}] " + " ".join(text.split())
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap
    return chunks

def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_context(query, chunks, index, top_k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    return "\n\n---\n\n".join([chunks[i] for i in indices[0] if i < len(chunks)])

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "simpler_answer" not in st.session_state:
    st.session_state.simpler_answer = None

with st.sidebar:
    st.header("1. Upload Chapter")
    uploaded_pdf = st.file_uploader("Choose a textbook chapter (PDF)", type=["pdf"])
    
    if uploaded_pdf is not None:
        if st.button("Process Chapter") or st.session_state.chunks is None:
            with st.spinner("Reading and indexing chapter..."):
                chunks = extract_and_chunk_pdf(uploaded_pdf)
                index = build_faiss_index(chunks)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.last_answer = None
                st.session_state.simpler_answer = None
            st.success(f"Indexed {len(chunks)} text chunks!")

if st.session_state.index is None:
    st.info("👈 Upload a textbook chapter PDF in the sidebar to begin.")
else:
    question = st.text_input("Ask your doubt from the chapter:", placeholder="e.g., What is deadlock in an operating system?")
    
    if st.button("Solve Doubt", type="primary") and question.strip():
        st.session_state.simpler_answer = None
        with st.spinner("Searching chapter and generating answer..."):
            context = retrieve_context(question, st.session_state.chunks, st.session_state.index)
            
            prompt = f"""You are a strict, helpful Textbook Doubt Solver. Answer the student's question using ONLY the provided textbook excerpts below. 

Rules:
1. Do NOT use outside knowledge. If the answer is not in the excerpts, say: "I could not find the answer to this in the uploaded textbook chapter."
2. Provide a clear, simple explanation in 2–4 sentences.
3. Include an exact, verbatim "Quoted Source Line" from the text that proves your answer.

Textbook Excerpts:
{context}

Student Question: {question}

Format your response strictly as:
**Explanation:** <your explanation>

**Quoted Source Line:** "<exact verbatim quote from the excerpts>"
"""
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
            )
            st.session_state.last_answer = response.choices[0].message.content

    if st.session_state.last_answer:
        st.markdown("### Answer")
        st.markdown(st.session_state.last_answer)
        
        if st.button("✨ Explain Simpler"):
            with st.spinner("Simplifying explanation..."):
                simpler_prompt = f"""Rewrite the following textbook explanation so a beginner student can easily understand it. Use an everyday analogy if helpful, keep it under 3 sentences, and do NOT add facts outside the original explanation.

Original Explanation:
{st.session_state.last_answer}
"""
                simpler_resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": simpler_prompt}],
                    model="openai/gpt-oss-120b",
                )
                st.session_state.simpler_answer = simpler_resp.choices[0].message.content

    if st.session_state.simpler_answer:
        st.markdown("### 💡 Simpler Explanation")
        st.success(st.session_state.simpler_answer)