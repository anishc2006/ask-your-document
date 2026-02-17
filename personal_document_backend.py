import os
import tempfile
import logging
import pandas as pd
from typing import Any, List, Dict
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY1 = os.getenv("GROQ_API_KEY1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an AI assistant that helps users understand and extract insights 
from their uploaded documents (PDF, Word, Excel, CSV, or text).
Answer questions clearly, concisely, and only based on the content of the documents. 
If the answer cannot be found, say so honestly.

Context:
{context}

Question: {question}
Answer:""",
)


def extract_text_from_pdf(pdf_path: str, max_pages: int = 2) -> str:
    """Extract text from a PDF file using PyPDF2 (no OCR)."""
    logger.info("Extracting PDF text: %s", pdf_path)
    text = ""
    try:
        reader = PdfReader(pdf_path)
        pages_to_read = min(max_pages, len(reader.pages))
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        logger.error("Failed to extract PDF text: %s", e)
    return text.strip()

def df_to_document(df: pd.DataFrame, source_name: str) -> Document:
    text = df.to_string(index=False)
    return Document(page_content=text, metadata={"source": source_name, "type": "dataframe"})

def load_files(uploaded_files: List[Any]) -> List[Document]:
    """Load uploaded files and return a list of Document objects (no OCR)."""
    docs: List[Document] = []

    for file in uploaded_files:
        filename = getattr(file, "name", "uploaded_file")
        suffix = os.path.splitext(filename)[1] or ""

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if filename.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(tmp_path)
                docs.append(df_to_document(df, filename))

            elif filename.lower().endswith(".csv"):
                df = pd.read_csv(tmp_path)
                docs.append(df_to_document(df, filename))

            elif filename.lower().endswith(".txt"):
                docs.extend(TextLoader(tmp_path, encoding="utf-8").load())

            elif filename.lower().endswith(".pdf"):
                pdf_text = extract_text_from_pdf(tmp_path, max_pages=10)
                docs.append(Document(page_content=pdf_text, metadata={"source": filename}))

            else:
                try:
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": filename}))
                except Exception:
                    docs.append(Document(page_content="", metadata={"source": filename, "note": "unreadable"}))
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                logger.debug("Could not remove temp file: %s", tmp_path)

    if not docs:
        raise ValueError("No valid documents loaded.")
    logger.info("Loaded %s documents.", len(docs))
    return docs

def build_qa_chain(docs: List[Document], k: int = 5, chunk_size: int = 500, chunk_overlap: int = 50) -> RetrievalQA:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    logger.info("Split into %s chunks.", len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.3, groq_api_key=GROQ_API_KEY1)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
    )
    logger.info("QA chain built successfully.")
    return qa_chain

def answer_query(qa_chain: RetrievalQA, query: str) -> Dict[str, Any]:
    response = qa_chain.invoke({"query": query})
    raw_result = response.get("result", response)
    source_docs = response.get("source_documents", [])
    return {
        "answer": raw_result,
        "sources": [doc.metadata for doc in source_docs],
    }

def summarize_documents(qa_chain: RetrievalQA, summary_query: str = "Summarize the key points of the uploaded documents.") -> str:
    try:
        result = qa_chain.invoke({"query": summary_query})
        return result.get("result", "No summary generated.")
    except Exception as e:
        logger.exception("Error while summarizing: %s", e)
        return f"Error while summarizing: {e}"

def build_pipeline_from_uploads(uploaded_files: List[Any]) -> RetrievalQA:
    """Convenience function: load files -> build QA chain."""
    docs = load_files(uploaded_files)
    qa = build_qa_chain(docs)
    return qa


