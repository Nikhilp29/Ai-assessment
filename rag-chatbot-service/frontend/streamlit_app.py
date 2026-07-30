"""
Streamlit frontend for the RAG Chatbot service. Thin client -- all parsing,
embedding, retrieval, and generation happens in the FastAPI backend.
"""
import os
import uuid
import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 RAG Chatbot")
st.caption("Upload documents, then ask questions answered from their content.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "sources": [...]}

# --- sidebar: backend status, document upload, reset ----------------------
with st.sidebar:
    st.subheader("Backend status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success("Connected")
        st.write(f"**LLM:** {health['llm_provider']}")
        st.write(f"**Embeddings:** {health['embedding_provider']}")
        st.write(f"**Chunks indexed:** {health['documents_indexed']}")
    except requests.exceptions.RequestException:
        st.error("Backend not reachable")
        st.code("uvicorn app.main:app --reload --port 8000")

    st.divider()
    st.subheader("Upload a document")
    uploaded_file = st.file_uploader("PDF or DOCX", type=["pdf", "docx"])
    if uploaded_file is not None and st.button("Ingest document"):
        with st.spinner("Chunking and embedding..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                r = requests.post(f"{API_URL}/ingest", files=files, timeout=120)
                if r.status_code == 200:
                    result = r.json()
                    st.success(
                        f"Indexed '{result['filename']}' — "
                        f"{result['chunks_added']} chunks added "
                        f"({result['total_chunks_in_store']} total in store)."
                    )
                else:
                    st.error(f"Ingest failed ({r.status_code}): {r.json().get('detail', r.text)}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")

    try:
        status = requests.get(f"{API_URL}/status", timeout=5).json()
        if status["indexed_files"]:
            st.write("**Indexed files:**")
            for f in status["indexed_files"]:
                st.write(f"- {f}")
    except requests.exceptions.RequestException:
        pass

    st.divider()
    if st.button("Reset conversation + documents"):
        try:
            requests.post(f"{API_URL}/reset", params={"session_id": st.session_state.session_id}, timeout=10)
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.success("Reset done.")
            st.rerun()
        except requests.exceptions.RequestException as exc:
            st.error(f"Reset failed: {exc}")

# --- chat history ----------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['filename']}** (distance: {s['distance']})")
                    st.caption(s["preview"] + "...")

# --- chat input -------------------------------------------------------------
user_input = st.chat_input("Ask a question about your uploaded documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": user_input},
                    timeout=120,
                )
                if r.status_code == 200:
                    result = r.json()
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("Sources used"):
                            for s in result["sources"]:
                                st.markdown(f"**{s['filename']}** (distance: {s['distance']})")
                                st.caption(s["preview"] + "...")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                    )
                else:
                    detail = r.json().get("detail", r.text)
                    st.error(f"Request failed ({r.status_code}): {detail}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ Error: {detail}", "sources": []}
                    )
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")
