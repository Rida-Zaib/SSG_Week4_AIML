# RAG Application — Architecture Notes

## What it does
Answers questions about Skill Set Go EduTech's AI/ML track by retrieving relevant
passages from a small knowledge base (`documents/company_faq.txt`) and passing them
to an LLM as context, instead of relying on the model's raw training data.

## Pipeline stages

1. **Load** — `DirectoryLoader` reads every `.txt` file in `documents/`.
2. **Split** — `RecursiveCharacterTextSplitter` breaks documents into ~400-character
   chunks with 50 characters of overlap, so no single chunk gets too large for the
   embedding model and context isn't lost at chunk boundaries.
3. **Embed** — each chunk is converted into a vector locally using
   `sentence-transformers/all-MiniLM-L6-v2`, no API key or cost involved.
4. **Store** — vectors are stored in a **FAISS** index, a fast local vector store
   for similarity search. The index is saved to disk (`faiss_index/`) so it doesn't
   need to be rebuilt every run.
5. **Retrieve** — given a question, the retriever embeds it and pulls back the top
   3 most similar chunks (`k=3`).
6. **Generate** — the retrieved chunks plus the question are sent to `openai/gpt-oss-20b`
   (running on Groq's free API) via LangChain's `RetrievalQA` chain, which produces
   a grounded answer and reports which source chunks it used.

## Fully free stack — no billing required
This pipeline uses **zero paid services**:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, run locally via
  HuggingFace — no API key, no cost, downloads once (~80MB) then works offline.
- **LLM**: Groq's free tier (no credit card) running `openai/gpt-oss-20b`, an
  open-weight model with full tool-calling and RAG support.

## Why FAISS over building this by hand
FAISS handles the nearest-neighbor search over embeddings efficiently, so retrieval
stays fast even as the document set grows. Reimplementing cosine similarity search
by hand would work at small scale but wouldn't hold up as more documents are added.

## Requirements
```
pip install -r ../requirements.txt
```
A `GROQ_API_KEY` environment variable must be set before running (free, no
credit card — get one at console.groq.com/keys).

## How to run
```bash
export GROQ_API_KEY="your-key-here"
python3 rag_pipeline.py
```
First run downloads the local embedding model (~80MB, one-time) and builds
the FAISS index; subsequent runs reuse both.

## Possible extensions
- Swap `DirectoryLoader` for a PDF or web loader to index other document types.
- Swap FAISS for Chroma with `langchain_community.vectorstores.Chroma` if a
  persistent client-server vector store is preferred over a local file index.
- Add a `ConversationalRetrievalChain` to support follow-up questions with memory.
