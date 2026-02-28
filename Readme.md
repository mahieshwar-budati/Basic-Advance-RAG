# Advanced RAG Implementation using LangChain + Milvus + FLAN-T5
📌 Project Overview
```
- This project implements a Retrieval-Augmented Generation (RAG) pipeline from scratch using:
- LangChain – Document processing & chunking
- UnstructuredPDFLoader – PDF parsing
- HuggingFace Embeddings – Vector generation
- Milvus – Vector database
- FLAN-T5 (Google) – LLM for answer generation
- PyTorch – Model execution
```
-> This is not a wrapper-based implementation.
-> It is a manual RAG architecture where each stage is clearly controlled.
---
## RAG
```
RAG = Retrieval + Augmentation + Generation
```
-> Instead of letting the LLM guess from its training data:
1. We retrieve relevant document chunks
2. We inject them into the prompt.
3. Then the LLM generates grounded answers
-> This reduces hallucination and improves accuracy.
