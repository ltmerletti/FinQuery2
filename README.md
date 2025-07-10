# FinQuery

```
▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖ ▗▖ ▗▖▗▄▄▄▖▗▄▄▖▗▖  ▗▖
▐▌     █  ▐▛▚▖▐▌▐▌ ▐▌ ▐▌ ▐▌▐▌   ▐▌ ▐▌▝▚▞▘  
▐▛▀▀▘  █  ▐▌ ▝▜▌▐▌ ▐▌ ▐▌ ▐▌▐▛▀▀▘▐▛▀▚▖ ▐▌  
▐▌   ▗▄█▄▖▐▌  ▐▌▐▙▄▟▙▖▝▚▄▞▘▐▙▄▄▖▐▌ ▐▌ ▐▌
```

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![React 19](https://img.shields.io/badge/React-19-blue?logo=react&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-blue?logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-latest-blue?logo=langchain)
![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-blue?logo=python&logoColor=yellow)

**An advanced, full-stack RAG application for querying complex financial documents using local-first AI models.**

FinQuery is a full-stack application designed for semantic search and question-answering on complex financial documents. It leverages a local-first Retrieval-Augmented Generation (RAG) pipeline, which can ensure privacy while maintaining performance. By combining multi-step parsing with a multi-stage retrieval process, FinQuery delivers accurate, context-aware answers from dense technical texts.

## Key Features

- **Local-First Architecture**: All AI processing happens on your machine with complete data privacy
- **Advanced RAG Techniques**: Multi-stage retrieval with semantic chunking, chunk augmentation, and cross-encoder reranking
- **Comprehensive Observability**: Full tracing with self-hosted Langfuse for debugging and optimization
- **Modular Monorepo Design**: Clean separation between reusable parsing library and main application
- **Financial Document Expertise**: Specialized parsing for complex financial documents with sophisticated table handling

## Tech Stack

### Backend
- **Language**: [Python 3.13](https://www.python.org/downloads/release/python-3130/)
- **API Framework**: [Flask](https://flask.palletsprojects.com/) 
- **AI Framework**: [LangChain](https://www.langchain.com/)
- **Observability**: [LangFuse](https://langfuse.com/)

### Frontend
- **Core**: [React 19](https://react.dev/blog/2024/04/25/react-19)
- **Build Tool**: [Vite](https://vitejs.dev/)

### Databases
- **Vector Store**: [ChromaDB](https://www.trychroma.com/)
- **Record Management**: [PostgreSQL](https://www.postgresql.org/)

### AI & Data Processing
- **PDF Parsing**: [Unstructured.io](https://unstructured.io/)
- **Embedding Model**: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **High-Fidelity Generation**: [qwen3-30b-a3b-dwq-053125](https://huggingface.co/mlx-community/Qwen3-30B-A3B-4bit-DWQ-053125)
- **Utility Generation**: [qwen3-30b-a3b-mixed-3](https://huggingface.co/mlx-community/Qwen3-30B-A3B-mixed-3-4bit) 
- **Reranking Model**: [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Active Python virtual environment
- System dependency: `tesseract-ocr`
- **LM Studio or compatible API server running locally** (for AI model inference)

## Quick Start

### 1. Clone & Setup Environment
```bash
git clone <your-repo-url>
cd FinQuery2

# Move and configure environment variables
mv docs/.env.example .env
# Edit .env with your configuration
```

### 2. Install Dependencies
```bash
# Activate your virtual environment
source .venv/bin/activate

# Install the parser library
pip install -e packages/finquery_parser

# Install the main application
pip install -e packages/finquery_app

# Install frontend dependencies
cd packages/finquery_frontend
npm install
```

### 3. Start Services

**Terminal 1: Langfuse**
```bash
docker-compose up -d
# Access at http://localhost:3000
```

**Terminal 2: Flask Backend**
```bash
python packages/finquery_app/src/finquery_app/api/main.py
# Access at http://localhost:5001
```

**Terminal 3: Frontend**
```bash
cd packages/finquery_frontend
npm run dev
```

## System Architecture

```mermaid
flowchart TD
    %% --- Ingestion Pipeline ---
    subgraph "One-Time Ingestion Pipeline"
        direction LR
        IngestStart(PDF Document) --> Parse["Custom Parsing (finquery_parser)"]
        Parse --> Chunk["Semantic Chunking (Text/Tables)"]
        Chunk --> Augment["Chunk Augmentation (Keywords, Summary)"]
        Augment --> Embed("Embedding Model")
        Embed --> Store(Vector Store & Record Manager)
    end

    %% --- Query & Observability Pipeline ---
    subgraph "Per-Query Retrieval Pipeline"
        A[User asks a question] --> D(Vector Store)
        D -- "Retrieves Top 10 Chunks" --> E{Reranking}
        E -- "Reranks for relevance" --> F[Top 4 Chunks]
        F -- "Relevant Context" --> G[Prompt Template]
        A -- "Original Query" --> G
        G --> H{Final LLM Call}
        H -- "Generates final answer" --> I[Answer + Source Metadata]
        
        subgraph "Observability"
            direction RL
            L(Langfuse)
            A -- "Trace" --> L
            E -- "Trace" --> L
            H -- "Trace" --> L
        end
    end

    %% --- Final Output ---
    I --> J[User receives answer with sources]
```

## How We Ensure Accurate Retrieval

### Custom Parsing Pipeline
- Custom LangChain component with specialized financial document parsing
- Unstructured's "hi-res" mode for maximum PDF fidelity
- Preprocessing removes repetitive elements (headers, footers, pagination)
- Tables preserved in structured markdown format

### Semantic Chunking Strategy
- Separate chunking approaches for text content versus tabular data
- Strategic overlap between chunks to preserve semantic context
- Chunk sizes optimized for both retrieval performance and context preservation

### Chunk Augmentation
- AI-generated metadata including relevance keywords and summaries
- Enhanced searchability through multiple representation vectors
- Page numbers, section titles, and document hierarchy preserved

### Cross-Encoder Reranking
- Qwen/Qwen3-Reranker-0.6B model reranks initial retrieval results
- Significant improvement in relevance ranking over semantic similarity alone
- Reduces noise from tangentially related content

## Data Flow

1. **Upload PDF** → Parse with AI → Store in encrypted DB
2. **User asks question** → AI understands and generates optimized query
3. **RAG retrieval** → Relevant information retrieved and reranked
4. **Final answer** → Generated with source tracing returned to user

## Project Structure

```
FinQuery2/
├── packages/
│   ├── finquery_parser/     # Reusable PDF parsing library
│   ├── finquery_app/        # Main Flask application
│   └── finquery_frontend/   # React frontend
├── chromadb/               # Vector database storage
├── reports/                # Sample documents
└── docs/                   # Documentation & schemas
```

This monorepo structure promotes code reuse and clear separation of concerns:
- **finquery_parser**: Self-contained, reusable library for PDF parsing
- **finquery_app**: Main Flask application with RAG implementation
- **finquery_frontend**: React-based user interface

## Future Roadmap

- [ ] **Advanced Table Parsing**: Implement dedicated table extraction libraries
- [ ] **Hybrid Chunking Strategy**: Formal separation of text vs table chunking
- [ ] **Multi-Representation Indexing**: Multiple vector representations per chunk
- [ ] **Evaluation Framework**: Further customize LangFuse for better observability
- [ ] **Query Transformation**: LLM translation layer for better keyword matching
- [ ] **Multi-Document Support**: Cross-document reasoning and comparison
- [ ] **User Chat Refinement**: Have a "chatting" AI to refine user question before querying

## Documentation

See the `docs/` folder for:
- Detailed flowcharts
- Prompt templates
- Database schemas
- API documentation
- Performance benchmarks

## Testing

```bash
# Run the test suite
cd packages/finquery_app/src/finquery_app/testing
pytest
```

## Design Philosophy

**Why RAG over Traditional Parsing?**

Financial documents come in non-standardized formats. While this project includes SEC filings, it's designed to be expandable for any type of financial document. RAG provides the flexibility to handle varied document structures while maintaining high accuracy.

**Local-First Approach**

Privacy is paramount when dealing with financial data. FinQuery runs entirely on your local machine, ensuring sensitive documents never leave your control while still providing enterprise-grade AI capabilities.

---

<details>
<summary>Research & Methodology</summary>

### Articles Used in Exploring and Improving RAG Methodology:
- **Multi-Representation**: https://towardsdatascience.com/multi-rep-colbert-retrieval-models-for-rags-fe05381b8819/
- **Chunking Considerations**: https://towardsdatascience.com/semantic-chunking-for-rag-35b7675ffafd/
- **Custom Pipelines**: https://towardsdatascience.com/callbacks-and-pipeline-structures-in-langchain-925aa077227e/
- **Query Transformation**: https://towardsdatascience.com/advanced-query-transformations-to-improve-rag-11adca9b19d1/
- **Chunk Augmentation**: https://x.com/svpino/status/1940006237384712404
- **Embedding Model Choice**: https://huggingface.co/spaces/mteb/leaderboard
- **Model Choice**: https://artificialanalysis.ai/
- **Best of 18 RAG Techniques**: https://levelup.gitconnected.com/testing-18-rag-techniques-to-find-the-best-094d166af27f#4630

### Why not use PDF Parsing and Relational Databases (Why Use RAG)?
The files will not all be in standardized formats; this project utilizes SEC filings, but it is made to be expandable, such that if someone were to upload other types of similar financial documents, they would easily be able to still use the tool.

</details>