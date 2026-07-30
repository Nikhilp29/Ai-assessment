"""
Streamlit frontend for the Document Summarization Service.

This is a thin client: all the real work (parsing, chunking, map-reduce,
LLM calls) happens in the FastAPI backend. This UI just uploads a file,
calls POST /summarize, and renders the response nicely.

Run the backend first (uvicorn app.main:app --reload --port 8000), then:
    streamlit run frontend/streamlit_app.py
"""
import os
import requests
import streamlit as st

API_URL = os.getenv("SUMMARIZER_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Document Summarization Service")
st.caption("Upload a PDF or DOCX and get a summary in your chosen style.")

# --- backend health check, shown in the sidebar -----------------------
with st.sidebar:
    st.subheader("Backend status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success("Connected")
        st.write(f"**Provider:** {health.get('provider')}")
        st.write(f"**Model:** {health.get('model')}")
    except requests.exceptions.RequestException:
        st.error("Backend not reachable")
        st.write(f"Tried: `{API_URL}/health`")
        st.write("Make sure the FastAPI server is running:")
        st.code("uvicorn app.main:app --reload --port 8000")

    st.divider()
    st.caption(f"API URL: `{API_URL}`")
    st.caption("Override with the SUMMARIZER_API_URL environment variable.")

# --- main form -----------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a PDF or DOCX file",
    type=["pdf", "docx"],
)

style = st.radio(
    "Summary style",
    options=["brief", "detailed", "bullet-points"],
    horizontal=True,
)

submit = st.button("Summarize", type="primary", disabled=uploaded_file is None)

if submit and uploaded_file is not None:
    with st.spinner("Summarizing… this can take a while for long documents (map-reduce runs one LLM call per chunk)."):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }
            data = {"style": style}
            response = requests.post(
                f"{API_URL}/summarize", files=files, data=data, timeout=300
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")
            st.stop()

    if response.status_code == 200:
        result = response.json()

        st.subheader("Summary")
        st.markdown(result["summary"])

        st.divider()
        cols = st.columns(4)
        cols[0].metric("Chunks", result["chunk_count"])
        cols[1].metric("Map-reduce used", "Yes" if result["used_map_reduce"] else "No")
        cols[2].metric("Time (s)", result["processing_time_seconds"])
        cols[3].metric("Model", result["model"])

        with st.expander("Raw JSON response"):
            st.json(result)

        st.download_button(
            "Download summary as .txt",
            data=result["summary"],
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_summary.txt",
        )
    else:
        # Surface the backend's error detail (400/422/429/502) rather than a generic failure
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"Request failed ({response.status_code}): {detail}")
