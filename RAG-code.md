# RAG-Code-Explanation

### Step 1 — Installing Dependencies
```
pip install sentence-transformers
pip install transformers
pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install unstructured
pip install pymilvus
```
->These libraries enable:
Library	                      - Purpose
sentence-transformers	        -Embedding model
transformers	                -FLAN-T5 model
langchain	                    -Document pipeline
unstructured	                -PDF parsing
pymilvus	                    -Vector DB connection
---
### Step 2 — Importing Required Modules
```
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Milvus
from pymilvus import connections
```
-> These imports define your RAG pipeline building blocks:
-Loader → Extract text
-Splitter → Create chunks
-Embeddings → Convert text → vectors
-Milvus → Store & retrieve vectors
---
### Step 3 — Loading the PDF
```
file_path = "/content/be.pdf"
loader = UnstructuredPDFLoader(file_path)
documents = loader.load()
```
What Happens Here:
-The PDF is parsed
-Each page becomes a document object
-Text content is extracted
->ou then combine all page contents:
```
all_text = []
for doc in documents:
    all_text.append(doc.page_content)
```
-> This creates one unified raw corpus.
---
### Step 4 — Chunking
```
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_text(full_text)
```
-> Why Chunking:
LLMs have token limits.

-> So we:
-Split text into 800-character chunks
-Add 100-character overlap
-Preserve context continuity

+Without overlap → context breaks
+Without chunking → token overflow

-> This is a crucial RAG optimization step.
---

### Step 5 — Embedding Model
```
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```
-> IT do:
- Each chunk is converted into a 384-dimensional dense vector.
- Now instead of storing text, we store semantic meaning.
- example:
```
"AI is revolutionizing healthcare"
→ [0.012, -0.88, 0.45, ...]
```
-> Vectors allow similarity search.
---
### Step 6 — Connecting to Milvus
```
connections.connect(
    alias="default",
    uri="./milvus_demo.db"
)
```
> -This creates a local Milvus instance.
Milvus stores:
- id (INT64 primary key)
- embedding (FLOAT_VECTOR dim=384)

Schema:
```
schema = CollectionSchema([
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=384)
])
```
-> Now your system has:
- Structured vector storage, Fast similarity search
---
### Step 7 — Inserting Embeddings into Milvus
-> When chunks are processed:
1. Each chunk → embedding
2. Embedding stored in Milvus collection
3. Collection indexed for similarity search
-> Milvus uses Approximate Nearest Neighbor (ANN) search internally.
---
### Step 8 — Loading FLAN-T5 LLM
```
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model_llm = T5F
ditionalGeneration.from_pretrained(model_name)
```
---
### Step 9 — Query Loop
```
while True:
    query = input("Ask question: ")
```
-> This creates a chatbot interface.
---
### Step 10 — Retrieval Phase
```
retrieved_chunks = (query, top_k=3)
```
-> What happens internally:
- Query → converted to embedding
- Milvus searches nearest vectors
- Top 3 most relevant chunks returned
-> This is semantic search, not keyword search.
---
### Step 11 — Prompt Augmentation
```
context = "\n".join([chunk[:500] for chunk in retrieved_chunks])
```
-> You inject retrieved content into prompt:
```
Use the provided context to answer clearly.
If not found, say you don't know.

Context:
{context}

Question:
{query}

Answer:
```
-> This is the Augmentation step. Now the LLM is grounded.
---
### Step 12 — Generation Phase
```
outputs = model_llm.generate(
    max_new_tokens=300,
    temperature=0.5,
    do_sample=False,
    repetition_penalty=1.1
)
```
-> Generation parameters:.
---
### Then decode:
```
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```
### Final output:
```
Answer: <Generated Response>
```

