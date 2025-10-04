# 📚 RAG Chatbot with Mistral 7B (Ollama + LangChain + Streamlit)

An intelligent Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, powered by **Mistral 7B** running locally through **Ollama**, and deployed via **Streamlit**. This bot answers questions from a custom book dataset (`yoursdocs.txt`) using semantic search and contextual understanding.

---

## 🚀 Features

- 🔍 **Semantic search over custom text documents** (e.g., book data, articles)
- 🧠 **Locally running Mistral 7B** LLM (no OpenAI key required)
- 🗂️ **Chunked document embeddings** using `SentenceTransformer` + `FAISS`
- ⚡ Fast and accurate **retrieval** using `RecursiveCharacterTextSplitter`
- 💬 Simple **chat interface** via Streamlit
- 💾 Optional **vectorstore caching** for faster reloads

---

## 🧱 Tech Stack

| Tool | Purpose |
|------|---------|
| [LangChain](https://github.com/langchain-ai/langchain) | RAG, document loading, chaining |
| [Ollama](https://ollama.com) | Runs Mistral 7B locally |
| [Mistral 7B](https://ollama.com/library/mistral) | Lightweight, high-performance LLM |
| [Streamlit](https://streamlit.io) | Web app UI |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector database |
| [Sentence Transformers](https://www.sbert.net/) | Document embeddings |

---

## 📁 Folder Structure

```
rag_mistral_app/
├── app.py                # Streamlit app file
├── data/
│   └── yoursdocs.txt     # Your custom knowledge base
├── faiss_index/          # (Auto-generated) vector store index
├── requirements.txt
├── README.md
```

---

## 📄 Setup Instructions

### 1. 🧠 Install Ollama and Pull Mistral

Download Ollama: https://ollama.com/download  
Then pull the Mistral model:

```bash
ollama pull mistral
```

Start the model:
```bash
ollama run mistral
```

---

### 2. 💻 Create Virtual Environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
```

---

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. 🏃‍♂️ Run the App

```bash
streamlit run app.py
```

Go to: [http://localhost:8501](http://localhost:8501)

---

## 📌 Sample Questions to Ask

- "Who is the author of *Malgudi Days*?"
- "Summarize the book *Karma* by Sadhguru."
- "Which books are in the comics genre?"
- "What is the description of *Black Holes* by Stephen Hawking?"

---

## 🧠 How It Works

1. Loads your `.txt` file with `TextLoader`
2. Splits text smartly using `RecursiveCharacterTextSplitter`
3. Embeds chunks using `all-MiniLM-L6-v2`
4. Stores/retrieves chunks with FAISS
5. Passes top results to Mistral via Ollama for answer generation

---

## ✅ Coming Soon (Suggestions)

- 🔄 Multi-file upload support (.pdf, .txt)
- 🧾 Chat history & session memory
- 💬 Voice/text input toggle
- 📚 Genre filters and book browsing
- ☁️ Switch to persistent DB like Chroma or Weaviate

---

## ✨ Credits

Built with ❤️ by [Jaya Lakshmi Katakam ](https://github.com/Jayakatakam)  
Inspired by modern RAG architecture & OSS LLMs.

---

## 📜 License

MIT License. Feel free to fork, use, or modify for educational or commercial use.

