# Sales RAG Assistant – High Precision Retrieval System

## Section 1: Context

This project implements a Retrieval-Augmented Generation (RAG) based Sales Assistant designed to answer business and sales-related questions from internal documents.

The primary objective of the system is to achieve **high-precision retrieval**, ensuring that the language model generates responses only from trusted document context instead of hallucinating unsupported information.

The workflow combines:
- Document ingestion
- Semantic chunking
- Vector embeddings
- Pinecone vector search
- LLM-based grounded response generation

The project demonstrates an understanding of modern AI application architecture using vector databases and retrieval pipelines.


---

# Section 2: Technical Implementation

## Architecture Flow

PDF Documents
↓
PyPDF Text Extraction
↓
RecursiveCharacterTextSplitter
↓
OpenAI Embeddings
↓
Pinecone Vector Database
↓
Similarity Search
↓
Retrieved Context
↓
LLM Response Generation

PDF → Text Chunking → OpenAI Embeddings → Pinecone Vector DB
                                         ↓
                                  Similarity Search
                                         ↓
                                  LLM Response


## Critical Retrieval Function

```python
def get_answer_from_docs(user_question, vector_store):

    # Retrieve top relevant chunks
    docs = vector_store.similarity_search(user_question, k=3)

    # Combine retrieved chunks into context
    context = "\n".join([doc.page_content for doc in docs])

    # Ground the LLM response
    prompt = f"""
    Use ONLY the following context to answer.
    If unsure, say 'I don't know'.

    Context:
    {context}
    """

    return llm.generate(prompt, user_question)
```

## Data Flow Explanation

1. PDF documents are loaded using PyPDF.
2. The extracted text is split into smaller semantic chunks using RecursiveCharacterTextSplitter.
3. Each chunk is converted into vector embeddings using OpenAI Embeddings.
4. Embeddings are stored in Pinecone for semantic similarity search.
5. User questions are converted into embeddings and matched against stored vectors.
6. The top relevant chunks are retrieved and passed to the LLM as grounded context.
7. The LLM generates a response strictly based on retrieved content.


---

# Section 3: Technical Decisions

## Why Pinecone instead of FAISS

I selected Pinecone instead of FAISS because Pinecone provides:
- Managed cloud infrastructure
- Better scalability
- Persistent vector storage
- Production-ready APIs
- Easier deployment for enterprise systems

FAISS is lightweight and efficient for local experimentation, but Pinecone is more suitable for scalable retrieval systems.

## Why Chunking Was Necessary

Chunking improves retrieval quality by:
- Preventing large context overflow
- Improving semantic relevance
- Increasing embedding accuracy
- Enabling fine-grained retrieval

## Hallucination Prevention Strategy

To reduce hallucinations:
- The LLM is instructed to answer only from retrieved context
- Retrieval is limited to top relevant chunks
- The prompt explicitly asks the model to say “I don’t know” if context is insufficient


---

# Section 4: Learning & Iteration

During development, I encountered multiple integration and dependency challenges related to modern LangChain modularization and Pinecone SDK migration.

One important learning was chunk sizing strategy. Initially, larger chunks reduced retrieval precision because unrelated information was grouped together. Smaller semantic chunks improved similarity matching and grounded response quality.

Additional learnings included:
- Managing Python virtual environments
- Handling package version conflicts
- Understanding embedding dimensionality
- Structuring modular AI pipelines
- Designing retrieval-first LLM systems


---

# Tech Stack

- Python
- LangChain
- OpenAI Embeddings
- Pinecone Vector Database
- PyPDF
- dotenv


---

# Folder Structure

sales-rag-assignment/

├── data/

│   └── sample.pdf

├── src/

│   ├── ingest.py

│   └── query.py

├── requirements.txt

├── .env

├── .gitignore

└── README.md


---

# Future Improvements

- Add Streamlit frontend
- Add conversational memory
- Add metadata filtering
- Implement hybrid search
- Add citation-based retrieval