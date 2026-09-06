# VectorNest

> A vector database built from scratch in Python, featuring custom nearest-neighbor indexes, persistent storage, semantic search, benchmarking, and RAG.

VectorNest is an educational, production-inspired vector database built to explore the internals of modern vector search systems.

Rather than relying on FAISS, Chroma, Pinecone, Weaviate, Qdrant, or another vector database for its core search engine, VectorNest implements vector storage, similarity metrics, metadata filtering, indexing, retrieval, and benchmarking itself.

It exposes these capabilities through a FastAPI REST API and an interactive web interface. Ollama is supported for fully local embeddings and generation, while Gemini can be used as an optional cloud AI provider for the deployed semantic-search and RAG demo.

## Live Demo

- **Web App:** https://vectornest.onrender.com
- **API Documentation:** https://vectornest-backend.onrender.com/docs

> The public demo runs on Render's free tier. Backend storage is therefore ephemeral and collections may be reset after service restarts or redeployments. Local and Docker deployments use VectorNest's persistent storage layer.

## Why VectorNest?

Vector databases are an important part of modern AI systems, especially semantic search and Retrieval-Augmented Generation (RAG).

Most applications interact with them as a black box.

VectorNest was built to understand what happens underneath that abstraction:

- How vectors are represented and validated
- How similarity between vectors is calculated
- How nearest-neighbor search works
- How indexing structures accelerate retrieval
- How metadata filtering interacts with vector search
- How documents become searchable embeddings
- How retrieved context can be passed to an LLM
- How approximate search quality can be measured using recall

The goal is not to replace production vector databases, but to build and understand their fundamental components through a clean, testable implementation.

---

## Features

### Vector Database Core

- Collection-based vector storage with fixed dimensions
- `float32` vector representation
- Vector and metadata validation
- Persistent JSON-backed storage
- Record insertion, retrieval, update, and deletion
- Exact-match metadata filtering

### Similarity & Distance Metrics

VectorNest implements:

- Cosine similarity
- Euclidean distance
- Dot product
- Manhattan distance

### Vector Indexes

Three search strategies are available:

- **Brute Force** — exact nearest-neighbor search
- **KD-Tree** — exact tree-based search for Euclidean distance
- **HNSW** — graph-based approximate nearest-neighbor search

Indexes are implemented as part of VectorNest rather than delegated to an external vector-search library.

### Semantic Search

VectorNest can:

1. Split documents into overlapping text chunks
2. Generate embeddings through a configurable embedding provider
3. Store vectors together with text and metadata
4. Embed natural-language queries
5. Retrieve semantically similar chunks using VectorNest's own indexes

Supported embedding providers:

- **Ollama** — local inference with `nomic-embed-text`
- **Gemini** — optional cloud inference using `gemini-embedding-2`

The default local configuration uses Ollama. The public deployment uses Gemini so that semantic search can run without depending on an Ollama instance on the user's machine.

### Retrieval-Augmented Generation

VectorNest includes a RAG pipeline:

`Question → Embedding → VectorNest Search → Retrieved Context → LLM → Answer`

Generation is provider-independent and currently supports:

- **Ollama** with `llama3.2` for fully local generation
- **Gemini** with `gemini-3.7-flash` for optional cloud generation

Both normal and streaming RAG responses are supported.

The external AI provider is used only for embedding and text generation. Vector storage, similarity computation, metadata filtering, indexing, and nearest-neighbor retrieval remain part of VectorNest itself.

### Benchmarking

The benchmarking layer compares vector indexes using:

- Index build time
- Query latency
- Recall@k

Brute-force search is used as the exact reference when evaluating approximate retrieval.

### Vector Visualization

The frontend includes a 2D vector projection view to visually explore stored vectors and query relationships.

### REST API

A FastAPI application exposes endpoints for:

- Collection management
- Vector record operations
- Vector search
- Metadata filtering
- Document ingestion
- Semantic search
- Vector projection
- RAG
- Streaming RAG

Interactive API documentation is available through FastAPI's Swagger interface.

### Docker Support

VectorNest includes containerized frontend and backend services with Docker Compose.

Persistent data is mounted outside the backend container so collections survive container recreation.

The Dockerized backend can communicate with a locally running Ollama instance for embeddings and generation.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Numerical Computing | NumPy |
| Backend API | FastAPI |
| ASGI Server | Uvicorn |
| Embeddings | Ollama (`nomic-embed-text`) / Gemini (`gemini-embedding-2`) |
| LLM | Ollama (`llama3.2`) / Gemini (`gemini-3.7-flash`) |
| Frontend | HTML, CSS, JavaScript |
| Static Web Server | Nginx |
| Testing | pytest |
| Linting | Ruff |
| Containerization | Docker & Docker Compose |
| Persistence | JSON-based local storage |
| Cloud Deployment | Render |

---

## High-Level Architecture

### Local / Docker Architecture
### Public Deployment Architecture

```text
┌─────────────────────────────┐
│       User's Browser        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Render Static Frontend   │
│      HTML / CSS / JS        │
└──────────────┬──────────────┘
               │ HTTPS
               ▼
┌─────────────────────────────┐
│    Render FastAPI Backend   │
└──────────────┬──────────────┘
               │
       ┌───────┴─────────┐
       │                 │
       ▼                 ▼
┌──────────────┐   ┌─────────────────┐
│  VectorNest  │   │ Gemini Provider │
│              │   │                 │
│ Storage      │   │ Embeddings      │
│ Filtering    │   │ Generation      │
│ Brute Force  │   └─────────────────┘
│ KD-Tree      │
│ HNSW         │
│ Metrics      │
└──────────────┘

```text
                        ┌─────────────────────┐
                        │      Frontend       │
                        │  HTML / CSS / JS    │
                        └──────────┬──────────┘
                                   │
                                   │ HTTP
                                   ▼
                        ┌─────────────────────┐
                        │     FastAPI API     │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
            ┌───────────────┐             ┌───────────────┐
            │ Service Layer │             │ Ollama        │
            │               │             │               │
            │ Search        │             │ Embeddings    │
            │ Ingestion     │◄───────────►│ LLM           │
            │ Semantic      │             └───────────────┘
            │ RAG           │
            │ Projection    │
            └───────┬───────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
 ┌────────────────┐   ┌────────────────┐
 │ Index Layer    │   │ Storage Layer  │
 │                │   │                │
 │ Brute Force    │   │ In-Memory      │
 │ KD-Tree        │   │ Persistent     │
 │ HNSW           │   │ Serialization  │
 └───────┬────────┘   └────────────────┘
         │
         ▼
 ┌────────────────┐
 │ Vector Metrics │
 │                │
 │ Cosine         │
 │ Euclidean      │
 │ Dot Product    │
 │ Manhattan      │
 └────────────────┘
```

---

## How VectorNest Works

VectorNest follows a layered architecture so that API handling, business logic, indexing, storage, and vector mathematics remain independent.

### 1. Vector Search Flow

When a vector search request reaches VectorNest:

```text
Client
  │
  ▼
FastAPI Search Endpoint
  │
  ▼
SearchService
  │
  ├── Metadata Filtering (optional)
  │
  ▼
Index Factory
  │
  ├── Brute Force
  ├── KD-Tree
  └── HNSW
  │
  ▼
Vector Metric
  │
  ├── Cosine Similarity
  ├── Euclidean Distance
  ├── Dot Product
  └── Manhattan Distance
  │
  ▼
Top-k Search Results
```

The API layer handles request/response validation while `SearchService` coordinates retrieval. Index implementations are separated behind a common abstraction, allowing different search strategies to be selected without changing the API layer.

---

### 2. Document Ingestion Flow

Raw text must be converted into vectors before semantic retrieval is possible.

```text
Document
   │
   ▼
Text Chunker
   │
   ▼
Overlapping Text Chunks
   │
   ▼
Ollama Embedding Provider
(nomic-embed-text)
   │
   ▼
768-dimensional Embeddings
   │
   ▼
VectorRecord
   │
   ├── Vector
   ├── Original Chunk
   └── Metadata
   │
   ▼
Persistent Storage
```

Chunk metadata preserves information such as the document identifier, chunk index, and text boundaries so retrieved vectors can be traced back to their source text.

---

### 3. Semantic Search Flow

Semantic search allows users to search using natural language instead of manually supplying vectors.

```text
Natural-Language Query
        │
        ▼
Embedding Provider
        │
        ▼
Query Vector
        │
        ▼
SemanticSearchService
        │
        ▼
Vector Search
        │
        ▼
Top-k Semantically Similar Chunks
```

The query and stored documents are embedded using the same embedding model, allowing their vector representations to be compared in the same embedding space.

---

### 4. RAG Flow

VectorNest builds Retrieval-Augmented Generation on top of semantic search.

```text
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Semantic Vector Search
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Build Context
      │
      ▼
Ollama LLM (llama3.2)
      │
      ▼
Grounded Answer
```

Instead of asking the language model to answer using only its internal knowledge, VectorNest retrieves relevant stored text and provides that context to the model.

The API supports both standard responses and NDJSON streaming, allowing generated tokens to be delivered incrementally to the frontend.

---

### 5. Persistence Flow

VectorNest separates the storage interface from its concrete implementations.

```text
Application
     │
     ▼
StorageBackend
     │
     ├── In-Memory Storage
     │
     └── Persistent Storage
              │
              ▼
       Serialized Collection
       and Record Data
              │
              ▼
       Local Filesystem
```

When running through Docker Compose, the persistent data directory is bind-mounted from the host into the backend container. This allows collections and records to survive container recreation.

---

## Internal Backend Structure

```text
vectornest/
│
├── api/            # FastAPI routes, schemas, dependencies and error handling
├── benchmarking/   # Index benchmarking and recall measurement
├── core/           # Configuration, shared types and exceptions
├── embeddings/     # Embedding provider abstraction and Ollama integration
├── indexes/        # Brute Force, KD-Tree and HNSW implementations
├── ingestion/      # Document chunking
├── metrics/        # Vector similarity and distance functions
├── models/         # Collection and vector record domain models
├── query/          # Metadata filtering
├── services/       # Search, ingestion, semantic search, RAG and projection
└── storage/        # In-memory/persistent storage and serialization
```

This separation keeps the core vector-search logic independent from the API and external AI services, making individual components easier to test and replace.

## Search Indexes

VectorNest implements three vector search strategies from scratch:

| Index | Type | Approach | Trade-off |
|---|---|---|---|
| **Brute Force** | Exact | Compares the query with every stored vector | Simple and accurate, but scales linearly |
| **KD-Tree** | Exact | Recursively partitions the vector space and prunes search branches | Effective in lower dimensions; degrades in high-dimensional spaces |
| **HNSW** | Approximate | Navigates a hierarchical proximity graph | Faster candidate exploration with a recall/speed trade-off |

Brute Force also acts as the exact reference implementation when benchmarking approximate search using **Recall@k**.

> **HNSW note:** VectorNest implements an educational HNSW with hierarchical random levels, bounded bidirectional connections, greedy upper-layer descent, and `ef_search` exploration. Neighbor selection during index construction is simplified compared with production HNSW implementations.

### Supported Vector Metrics

- Cosine similarity
- Euclidean distance
- Dot product
- Manhattan distance

Metadata can also be attached to vectors and used for exact-match filtering alongside vector search.

## Quick Start

### Prerequisites

Make sure you have installed:

- Docker Desktop
- Ollama
- Git

Pull the models used by VectorNest:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Run with Docker

Clone the repository:

```bash
git clone https://github.com/adityaraghav1/VectorNest.git
cd VectorNest
```

Start the frontend and backend:

```bash
docker compose up --build
```

Once the containers are running:

| Service | URL |
|---|---|
| Frontend | `http://127.0.0.1:5500` |
| Backend API | `http://127.0.0.1:8000` |
| Swagger API Docs | `http://127.0.0.1:8000/docs` |

Ollama runs on the host machine and is accessed by the backend container through `host.docker.internal`.

Persistent VectorNest data is stored under:

```text
data/vectornest/
```

Stop the application with:

```bash
docker compose down
```

Container recreation does not delete stored collections because the data directory is mounted from the host.

---

## Basic Workflow

Using the web interface:

```text
Create Collection
      ↓
Ingest Document
      ↓
Chunk + Embed Text
      ↓
Store Vectors
      ↓
Semantic Search
      ↓
Retrieve Relevant Context
      ↓
Generate RAG Answer
```

For the default Ollama embedding model, create collections with a dimension of **768**.

## Testing

VectorNest includes automated tests across the core database, API, indexing, persistence, embeddings, ingestion, semantic search, benchmarking, and RAG layers.

Run the backend test suite:

```bash
cd backend
python -m pytest
```

Run linting:

```bash
python -m ruff check .
```

The test suite covers components including:

- Vector metrics and validation
- Brute Force, KD-Tree, and HNSW indexes
- Storage and serialization
- Metadata filtering
- Search and benchmarking
- Document chunking and ingestion
- Ollama embedding integration
- Semantic search and RAG
- REST API behavior and error handling

---

## Project Structure

```text
VectorNest/
├── backend/
│   ├── vectornest/
│   │   ├── api/           # FastAPI routes and schemas
│   │   ├── benchmarking/  # Performance and recall evaluation
│   │   ├── core/          # Configuration, types and exceptions
│   │   ├── embeddings/    # Embedding provider integrations
│   │   ├── indexes/       # Brute Force, KD-Tree and HNSW
│   │   ├── ingestion/     # Text chunking
│   │   ├── metrics/       # Vector distance/similarity functions
│   │   ├── models/        # Collection and record models
│   │   ├── query/         # Metadata filtering
│   │   ├── services/      # Application/service layer
│   │   └── storage/       # Storage and serialization
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── Dockerfile
│
├── data/                  # Local persistent database data
├── compose.yaml
└── README.md
```

---

## Current Limitations

VectorNest is designed as an educational, production-inspired system rather than a replacement for production vector databases.

Current limitations include:

- HNSW uses a simplified neighbor-selection strategy during index construction.
- Persistent storage is file-based and is not designed for concurrent multi-process writes.
- Document ingestion is not transactional, so a failure during ingestion can leave previously inserted chunks stored.
- RAG conversations are stateless at the backend level.
- Local AI functionality depends on Ollama and locally available models.
- The current implementation focuses on correctness, architecture, and algorithm exploration rather than distributed scaling.

These trade-offs are intentionally documented to distinguish the project's implemented behavior from production-scale vector database guarantees.

---

## Future Improvements

Potential extensions include:

- More production-like HNSW construction
- Atomic document ingestion
- Concurrent storage handling
- Index persistence
- Hybrid lexical + vector search
- Reranking retrieved results
- RAG conversation persistence
- Authentication and rate limiting
- Larger-scale performance benchmarking
- Cloud deployment

---

## License

This project is intended for educational and portfolio use.

## Project Status

VectorNest currently includes the complete core vector-search engine, persistent storage, indexing strategies, metadata filtering, benchmarking, REST APIs, semantic document ingestion, local RAG, visualization, automated tests, and Dockerized frontend/backend services.

The project is actively being finalized for deployment and further performance evaluation.