# 📌Embedding, Semantic Search and Local LLM with Ollama & FAISS

## 📖 Overview
This repository contains three practical exercises focused on using open-source language models and local embedding techniques to build semantic search systems. The exercises include:

1. Running LLaMA 3.1 using Ollama on Google Colab and prompting it with LangChain  
2. Building a local embedding pipeline for Persian text using ParsBERT  
3. Performing filtered vector search using metadata in FAISS

🧠 Features  
- Run `llama3:7b` with Ollama in Google Colab  
- Use `LangChain` to prompt locally served LLMs  
- Use `ParsBERT` for embedding Persian documents locally  
- Build a semantic search vector database with FAISS  
- Perform metadata-based filtering in vector search  
- Fully offline, private and language-specific NLP

📁 File Structure  
- `exercise_1_ollama_colab.ipynb` : Setting up and running LLaMA 3.1 using Ollama in Colab, and prompting it with LangChain  
- `exercise_2_local_embedding_faiss.ipynb` : Local embedding using `ParsBERT` + FAISS-based vectorstore for Persian texts  
- `exercise_3_metadata_filtering.ipynb` : Vector search with filtering based on metadata using FAISS and LangChain

## 🚀 How to Use  

### 🔧 Setup (General)

Install required packages:

```bash
pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers
```

✅ Exercise 1 – Running LLaMA3.1 on Google Colab with Ollama

1. Change Colab runtime to T4 GPU (optional but recommended)

2. Install Ollama and serve it in background:

```bash
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve &
```

3. Pull the LLaMA model:

```bash
!ollama run llama3:7b
```

4. After model is downloaded, type /bye to release the running cell

5. Use langchain_ollama.ChatOllama() to create an instance and interact with the model



✅ Exercise 2 – Local Embedding with ParsBERT + FAISS

1. Install required libraries:

```bash
pip install langchain langchain-huggingface langchain-community faiss-cpu
```

2. Load the model:

```bash
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

model_name = "HooshvareLab/bert-base-parsbert-uncased"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

hf_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
```
3. Embed a sample Persian text and build a FAISS index:

```bash
embed = hf_embedding.embed_query("سلام به هوشواره خوش آمدید")
```

4. Create vectorstore and add documents to it.


✅ Exercise 3 – Filtered Vector Search with Metadata

1. Load multiple documents (e.g., Wikipedia, PDFs, or text files)

2. Split into chunks (~500 characters) and assign metadata per source:

```bash
from langchain.docstore.document import Document

doc = Document(page_content="...", metadata={"source": "source_01"})
```

3. Build a FAISS vectorstore with all documents

4. Run filtered semantic search:

```bash
results = vector_store.similarity_search("your query", k=2, filter={"source": "source_01"})
```

### ✨ Key Features:

✔ Run open-source models (LLaMA 3.1) locally using Ollama in Colab

✔ Use Persian-specific embedding models for localized NLP

✔ Build scalable and efficient FAISS vectorstores

✔ Perform smart search with metadata filtering

✔ End-to-end pipeline for hybrid semantic search systems


### 📄 Sample Documents

To reproduce the experiments exactly as shown in the notebooks, please download the following files from this repository:

- attention.pdf

- Cognitive computational neuroscience.pdf

- Cognitive Neuroscience.pdf

- Proposal_erfan_english.docx

These documents are used as example inputs for demonstrating document loading, chunking, embedding generation, and metadata-based vector search.

Note:
You may freely replace these files with your own documents. The pipeline is domain-agnostic and will work with any PDF or DOCX files following the same structure.

