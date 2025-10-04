# import streamlit as st
# from langchain.llms import Ollama
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA

# # Page title
# st.set_page_config(page_title="📚 BookBot - Ask about any Book", layout="wide")
# st.title("📚 Ask Me Anything About the Book Collection")

# # User input
# query = st.text_input("🔍 Enter your question about any book, author, or topic:")

# # Load the uploaded .txt file
# loader = TextLoader("C:\Shashank\Ollama\rag_mistral_app\data\yoursdocs.txt")  # Ensure this file is in the same folder
# documents = loader.load()

# # Split the long document into manageable chunks
# splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs = splitter.split_documents(documents)

# # Convert to embeddings
# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = FAISS.from_documents(docs, embedding)

# # Setup retriever
# retriever = vectordb.as_retriever()

# # Load Mistral via Ollama
# llm = Ollama(model="mistral")

# # Build QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True
# )

# # Run query
# if query:
#     with st.spinner("Thinking..."):
#         result = qa_chain({"query": query})

#         st.subheader("🧠 Answer:")
#         st.write(result["result"])

#         st.subheader("📄 Sources:")
#         for i, doc in enumerate(result["source_documents"]):
#             st.markdown(f"**Source {i+1}:**")
#             st.markdown(doc.page_content)
#             st.markdown("---")
import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="📚 RAG Chatbot with Mistral 7B", layout="wide")
st.title("📚 Ask Me Anything from Your Book Dataset")

# ---------------------------
# File Path Handling (Windows-safe)
# ---------------------------
file_path = os.path.join("data", "yoursdocs.txt")

if not os.path.exists(file_path):
    st.error(f"❌ File not found at path: {os.path.abspath(file_path)}")
    st.stop()
else:
    st.success(f"✅ File found: {file_path}")

# ---------------------------
# Load Text File as Documents
# ---------------------------
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# ---------------------------
# Split into Chunks
# ---------------------------
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# ---------------------------
# Embeddings + FAISS Vectorstore
# ---------------------------
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# ---------------------------
# Mistral LLM via Ollama
# ---------------------------
llm = Ollama(model="mistral")

# ---------------------------
# RAG Chain Setup
# ---------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ---------------------------
# User Input
# ---------------------------
query = st.text_input("🔍 Ask your question:")

if query:
    with st.spinner("💡 Thinking..."):
        result = qa_chain({"query": query})

    # Answer
    st.subheader("🧠 Answer:")
    st.write(result["result"])

    # Sources
    st.subheader("📄 Source Passages:")
    for i, doc in enumerate(result["source_documents"]):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(doc.page_content)
        st.markdown("---")
