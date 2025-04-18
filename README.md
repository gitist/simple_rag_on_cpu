# Simple RAG Application on CPU

In order to run the code, follow these recommended steps. In our environment, we had Python 3.12 installed.

## Steps to Run the Code

1. **Clone the repository:**
    ```sh
    git clone https://github.com/gitist/simple_rag_on_cpu.git
    ```

2. **Create a Python virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment and install dependencies:**
    ```sh
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4. **Run the RAG app**
    ```sh
    uvicorn app:app
    ```


To interact with the app, open your browser and visit:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Workflow Diagram
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
