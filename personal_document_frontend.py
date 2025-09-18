import streamlit as st
import pandas as pd
import tempfile
from PyPDF2 import PdfReader
from personal_document_backend import load_files, build_qa_chain, answer_query, summarize_documents, extract_text_from_pdf

st.set_page_config(page_title="AI Document Q&A", layout="wide")

# Detect Streamlit theme (light or dark)
theme = st.get_option("theme.base")

if theme == "light":
    # Light mode CSS
    st.markdown(
        """
        <style>
            .stApp { background:#ffffff; color:#000000; font-family: 'Segoe UI', sans-serif; }
            .title { text-align: center; font-size: 32px; font-weight: 500; color: #000000; margin-bottom: 10px; }
            .panel { background:#f3f4f6; border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid rgba(0, 0, 0, 0.2); }
            section[data-testid="stSidebar"] { background:#f1f5f9; color: #000000; border-right: 1px solid #d1d5db; }
            section[data-testid="stSidebar"] h2 { color: #FFD700 !important; }
            h2, h3 { color: #000000 !important; padding-bottom: 4px; }
            .chat-box { max-height: 500px; overflow-y: auto; padding: 10px; }
            .chat-user { background:#e0f2fe; color:#000000; padding: 10px 14px; border-radius: 16px; margin: 4px 2px; text-align: left; max-width: 70%; margin-right: auto; white-space: pre-wrap; }
            .chat-ai { background:#f0fdf4; color:#000000; padding: 10px 14px; border-radius: 16px; margin: 4px 2px; text-align: left; max-width: 70%; margin-left: auto; white-space: pre-wrap; }
            div.stButton > button { background-color: #FFD700; color: black; font-weight: 500; border-radius: 6px; border: none; padding: 6px 14px; }
            div.stButton > button:hover { background-color: #fbbf24; color: blue; }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    # Dark mode CSS
    st.markdown(
        """
        <style>
            .stApp { background:#012624; color: #e5e7eb; font-family: 'Segoe UI', sans-serif; }
            .title { text-align: center; font-size: 32px; font-weight: 500; color: #f9fafb; margin-bottom: 10px; }
            .panel { background:#052626; border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid rgba(255, 215, 0, 0.2); }
            section[data-testid="stSidebar"] { background:#011c1c; color: #f1f5f9; border-right: 1px solid #2d3748; }
            section[data-testid="stSidebar"] h2 { color: #FFD700 !important; }
            h2, h3 { color: #fafafa !important; padding-bottom: 4px; }
            .chat-box { max-height: 500px; overflow-y: auto; padding: 10px; }
            .chat-user { background:#064663; color:#ffffff; padding: 10px 14px; border-radius: 16px; margin: 4px 2px; text-align: left; max-width: 70%; margin-right: auto; white-space: pre-wrap; }
            .chat-ai { background:#1b4332; color:#f1f5f9; padding: 10px 14px; border-radius: 16px; margin: 4px 2px; text-align: left; max-width: 70%; margin-left: auto; white-space: pre-wrap; }
            div.stButton > button { background-color: #FFD700; color: black; font-weight: 500; border-radius: 6px; border: none; padding: 6px 14px; }
            div.stButton > button:hover { background-color: #fbbf24; color: blue; }
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown("<h1 class='title'>“ASK YOUR DOCUMENT”</h1>", unsafe_allow_html=True)
with st.sidebar:
    st.title("DASHBOARD")
    st.header("📂 Upload Documents Here")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        if st.button("⚡ Build Knowledge Base"):
            with st.spinner("Processing files..."):
                try:
                    docs = load_files(uploaded_files)
                    qa_chain = build_qa_chain(docs)
                    st.session_state.qa_chain = qa_chain
                    st.session_state.docs = docs 
                    st.success("✅ Knowledge base is ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

tabs = st.tabs(["📄 Documents", "💬 Q&A", "📝 Summarize"])

with tabs[0]:
    st.subheader("Uploaded Document Preview")
    if uploaded_files:
        for file in uploaded_files:
            st.markdown(f"<div class='panel'><b>{file.name}</b></div>", unsafe_allow_html=True)
            file.seek(0)

            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                st.dataframe(df.head(20))

            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
                st.dataframe(df.head(20))

            elif file.name.endswith(".txt"):
                text = file.read().decode("utf-8").splitlines()
                preview = "\n".join(text[:30])
                st.text_area("Preview", preview, height=400)

            elif file.name.endswith(".pdf"):
                # Extract text from PDF without OCR
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    preview_text = extract_text_from_pdf(tmp.name, max_pages=2)
                st.text_area("Preview", preview_text[:4000], height=600)

            else:
                st.info("Preview not available for this file type.")
    else:
        st.info("Upload documents to see a preview.")

with tabs[1]:
    if "qa_chain" in st.session_state:
        st.subheader("Ask Your Questions")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        user_query = st.chat_input("Type your question...")
        if user_query:
            with st.spinner("Thinking..."):
                result = answer_query(st.session_state.qa_chain, user_query)
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": result["answer"],
                    "sources": result.get("sources", []),
                })
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"<div class='chat-user'>{chat['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-ai'>{chat['answer']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please upload and build the knowledge base first.")
with tabs[2]:
    if "qa_chain" in st.session_state:
        if st.button("📝 Summarize Documents"):
            with st.spinner("Generating summary..."):
                summary = summarize_documents(st.session_state.qa_chain)
                st.markdown("📌 Document Summary")
                for line in summary.split("\n"):
                    if line.strip():
                        st.markdown(f"- {line}")
    else:
        st.warning("⚠️ Please upload and build the knowledge base first.")
