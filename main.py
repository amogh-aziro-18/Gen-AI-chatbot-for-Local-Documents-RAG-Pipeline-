import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from typing import List
import re
import pypdf

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ------------------ Streamlit Configuration ------------------
st.set_page_config(page_title="🤖 RAG Chatbot", page_icon="🤖", layout="wide")

# ------------------ Custom Styling ------------------
st.markdown("""
<style>
    /* Fonts & Layout */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        color: #1a1a1a;
    }

    /* Chat message styling */
    div[data-testid="stChatMessage"] {
        padding: 1rem 1.2rem;
        border-radius: 1.2rem;
        margin-bottom: 0.8rem;
        max-width: 80%;
    }
    div[data-testid="stChatMessage"][data-testid="stChatMessage-user"] {
        background: linear-gradient(135deg, #b3e5fc 0%, #81d4fa 100%);
        color: #0d47a1;
        align-self: flex-end;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #1b5e20;
        margin-right: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background-color: #f4f6f8;
        border-right: 1px solid #dcdcdc;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2196f3, #21cbf3) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none;
        font-weight: 600;
    }

    /* Dark mode */
    body.dark-mode {
        background-color: #1e1e1e;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ App Header ------------------
st.title("🤖 Smart RAG Chatbot")
st.caption("💬 Ask questions about your PDF — powered by Groq + LangChain + LLaMA ⚡")

# ------------------ Theme Toggle ------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(f"**Current Theme:** `{st.session_state.theme.capitalize()}`")
with col2:
    st.button("🌓 Toggle Theme", on_click=toggle_theme)

# ------------------ Custom Dark Mode CSS ------------------
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
        body, .stApp {
            background-color: #121212 !important;
            color: #fafafa !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #1f1f1f !important;
            color: white;
        }
        div[data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] {
            background: linear-gradient(135deg, #263238 0%, #37474f 100%);
            color: #e0e0e0;
        }
        div[data-testid="stChatMessage"][data-testid="stChatMessage-user"] {
            background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
            color: #e3f2fd;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------ Custom Retriever ------------------
class ImprovedRetriever(BaseRetriever):
    documents: List[Document]
    k: int = 6
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scores = []
        
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            exact_match_score = 100 if query_lower in content_lower else 0
            content_words = set(content_lower.split())
            word_overlap = len(query_words.intersection(content_words))
            keyword_count = sum(content_lower.count(word) for word in query_words)
            metadata_boost = 50 if doc.metadata.get("type") == "metadata" else 0
            total_score = exact_match_score + (word_overlap * 10) + keyword_count + metadata_boost
            scores.append((doc, total_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:self.k]]

# ------------------ PDF Utilities ------------------
def load_pdf_fast(file_path: str, max_pages: int = 50):
    documents = []
    pdf_title = "Unknown Document"
    try:
        with open(file_path, 'rb') as file:
            pdf = pypdf.PdfReader(file)
            total_pages = min(len(pdf.pages), max_pages)
            if pdf.metadata and pdf.metadata.title:
                pdf_title = pdf.metadata.title
            for i in range(total_pages):
                try:
                    text = pdf.pages[i].extract_text()
                    if text and len(text.strip()) > 50:
                        documents.append(Document(page_content=text, metadata={"page": i+1, "source": pdf_title}))
                except:
                    continue
    except Exception as e:
        st.error(f"⚠️ PDF reading error: {str(e)}")
    return documents, pdf_title

def simple_split(documents: List[Document], chunk_size: int = 800) -> List[Document]:
    chunks = []
    for doc in documents:
        text = doc.page_content
        sentences = re.split(r'[.!?]\s+', text)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
                current_chunk = sentence + ". "
        if current_chunk.strip():
            chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
    return chunks

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("🔑 Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    max_pages = st.number_input("📄 Max pages", 1, 200, 50, 5)
    uploaded = st.file_uploader("📂 Upload your PDF", type=['pdf'])
    model = st.selectbox("🧠 Model", ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"])
    if uploaded and st.button("🚀 Process PDF"):
        st.session_state.process_trigger = True
        st.session_state.uploaded_file = uploaded

# ------------------ Initialize Session ------------------
for key in ['msgs', 'retriever', 'all_chunks', 'pdf_title', 'process_trigger', 'processing_done']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'retriever' else []

# ------------------ PDF Processing ------------------
if st.session_state.process_trigger and not st.session_state.processing_done:
    st.session_state.process_trigger = False
    with st.spinner("📖 Processing PDF..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
                f.write(st.session_state.uploaded_file.read())
                path = f.name
            docs, pdf_title = load_pdf_fast(path, max_pages)
            chunks = simple_split(docs)
            st.session_state.retriever = ImprovedRetriever(documents=chunks)
            st.session_state.pdf_title = pdf_title
            st.session_state.processing_done = True
            os.unlink(path)
            st.success(f"✅ Processed: {pdf_title}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ------------------ Chat Interface ------------------
for msg in st.session_state.msgs or []:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("💬 Ask something about your document..."):
    if not api_key:
        st.error("⚠️ Please enter your Groq API key!")
        st.stop()
    if not st.session_state.retriever:
        st.error("⚠️ Upload and process a PDF first.")
        st.stop()
    
    st.session_state.msgs.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=model,
                temperature=0.2,
                max_tokens=1000
            )
            prompt = PromptTemplate(
                template="""You are a helpful assistant. Use the provided context to answer the user's question accurately.
Context:
{context}

Question:
{question}

Answer:""",
                input_variables=["context", "question"]
            )
            chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff",
                retriever=st.session_state.retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=False
            )
            result = chain.invoke({"query": question})
            answer = result['result']
            st.write(answer)
            st.session_state.msgs.append({"role": "assistant", "content": answer})

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("✨ Built with ❤️ using Streamlit + LangChain + Groq | Enhanced RAG Chatbot")
