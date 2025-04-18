# simple_rag_on_cpu
Simple RAG running on laptop
```mermaid
flowchart TD;
    A[User sends POST /ask with query] --> B[Encode query with SentenceTransformer]
    B --> C[Search FAISS index for top_k similar docs]
    C --> D[Retrieve top_k documents]
    D --> E[Construct prompt]
    E --> F[Tokenize prompt with Flan-T5 tokenizer]
    F --> G[Generate answer using Flan-T5 model]
    G --> H[Decode and return answer to user]

    subgraph Initialization [App Startup]
        I[Load documents from file]
        J[Encode documents with SentenceTransformer]
        K[Build FAISS index with embeddings]
    end

    I --> J --> K
