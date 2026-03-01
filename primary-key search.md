# Milvus Vector Database – Primary Key Search Explained

### This document explains how Milvus performs:
- Collection creation.
-  Schema definition.
-  Vector indexing.
-  Data insertion.
-  Primary-key-based retrieval.
-  Similarity search.
-  This example demonstrates the core mechanics behind vector retrieval systems, which are foundational for Retrieval-Augmented Generation (RAG) architectures.
---
### 1. Import Required Modules
```
from pymilvus import MilvusClient, DataType
```
Explanation.
- MilvusClient → Main interface to interact with Milvus.
- DataType → Used to define schema field types such as INT64 and FLOAT_VECTOR.
- This is similar to defining table structure in traditional SQL databases.

### 2. Connect to Milvus (Lite Mode)
```
client = MilvusClient(uri="./milvus_demo.db")
```
What Happens Here?
- Creates a local Milvus database file.
- No server or Docker required.
- Ideal for development and experimentation.
- In production environments, Milvus typically runs as a service (e.g., on port 19530).

 ### 3. Drop Existing Collection (Safe Re-run)
 ```
if client.has_collection("products"):
    client.drop_collection("products")
```
Why This Step?
- Prevents schema conflicts.
- Ensures clean execution every time.
- Avoids duplicate index or structure errors.
- This is similar to resetting a table before recreating it.

### 4. Define Collection Schema
```
schema = client.create_schema(auto_id=True)
Add Primary Key Field
schema.add_field("id", DataType.INT64, is_primary=True)
```
- INT64 → 64-bit integer
- is_primary=True → Unique identifier
- auto_id=True → Automatically generated

### Equivalent SQL concept:
```
id BIGINT PRIMARY KEY AUTO_INCREMENT
Add Vector Field
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=5)
```
- FLOAT_VECTOR → Stores embedding vectors
- dim=5 → Each vector has 5 dimensions (demo purpose)
```
 Example stored vector:

[0.1, 0.2, 0.3, 0.4, 0.5]
```

-> In real-world RAG systems:
- Sentence Transformers → 384 dimensions
- OpenAI embeddings → 1536 dimensions

### 5.  Create Index for Fast Similarity Search
```
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="L2"
)
```

-> Indexing Is Important
- Without indexing:
- Milvus compares every vector
- Slow for large datasets

-> With indexing:
- Uses Approximate Nearest Neighbor (ANN)
- Fast similarity retrieval
- Distance Metric: L2
-  metric_type="L2" uses Euclidean Distance:
(𝑥1-y1)2+(𝑥2−𝑦2)2/(x1−y1)2+(x2−y2)2
- Other common metrics:
- COSINE
- IP (Inner Product)
- For RAG systems, cosine similarity is commonly used.

 ### 6. Create the Collection
```
client.create_collection(
    collection_name="products",
    schema=schema,
    index_params=index_params
)
```
###
Now the collection structure is:
```
Collection: products
 ├── id (INT64, Primary Key)
 └── vector (FLOAT_VECTOR, dim=5)
```
### 7. Insert Sample Data
```
data = [
    {"vector": [0.1, 0.2, 0.3, 0.4, 0.5]},
    {"vector": [0.2, 0.3, 0.4, 0.5, 0.6]},
    {"vector": [0.9, 0.8, 0.7, 0.6, 0.5]}
]

insert_result = client.insert(
    collection_name="products",
    data=data
)
```
-> What Happens Internally?
- Each vector is stored as a record.
- Milvus auto-generates unique IDs.
- Insert result returns generated primary keys.
- Example:
insert_result["ids"]
-> These IDs are later used for retrieval.

### 8. Primary-Key-Based Search
- Instead of directly searching using a random vector, we:
- Select an existing ID
- Retrieve its vector
- Use that vector for similarity search

#### Step 1 — Select Existing ID
```
query_id = insert_result["ids"][0]
```
-> This picks the first inserted record.

#### Step 2 — Retrieve That Record
```
entity = client.query(
    collection_name="products",
    filter=f"id == {query_id}",
    output_fields=["vector"]
)
```

-> This fetches the vector associated with the selected ID.
- query_vector = entity[0]["vector"]
- Now we have the exact stored embedding.

#### Step 3 — Perform Similarity Search
```
results = client.search(
    collection_name="products",
    data=[query_vector],
    limit=2
)
```
-> What Happens Here:
- Milvus compares query_vector with all stored vectors.
- Uses L2 distance.
- Returns top 2 closest matches.
