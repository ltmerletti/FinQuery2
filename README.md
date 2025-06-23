```
▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖ ▗▖ ▗▖▗▄▄▄▖▗▄▄▖▗▖  ▗▖
▐▌     █  ▐▛▚▖▐▌▐▌ ▐▌ ▐▌ ▐▌▐▌   ▐▌ ▐▌▝▚▞▘  
▐▛▀▀▘  █  ▐▌ ▝▜▌▐▌ ▐▌ ▐▌ ▐▌▐▛▀▀▘▐▛▀▚▖ ▐▌  
▐▌   ▗▄█▄▖▐▌  ▐▌▐▙▄▟▙▖▝▚▄▞▘▐▙▄▄▖▐▌ ▐▌ ▐▌                                      
```

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![React 19.1.0](https://img.shields.io/badge/React-19.1.0-blue?logo=react)
![ChromaDB 1.0.13](https://img.shields.io/badge/ChromaDB-1.0.13-blue?logo=python)

### Key Steps:

1. Upload PDF → Parse with AI → Store in encrypted DB
2. User asks question → AI understands and generates math query
3. RAG used to retrieve information, given to user
4. Final answer with trace is returned to user

---

### Backend

- **Primary Language**: [Python](https://www.python.org/)
- **API Framework**: [Flask](https://flask.palletsprojects.com/)
- **Database ORM**: [SQLAlchemy](https://www.sqlalchemy.org/)
- **Testing Framework**: [pytest](https://docs.pytest.org/)

### Frontend

- **Core Library**: [React JS](https://react.dev/)
- **Build Tool**: [Vite](https://vitejs.dev/)

### Databases

- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Relational Database**: [PostgreSQL](https://www.postgresql.org/)

### AI & Data Processing

- **AI Control Framework**: [LangChain](https://www.langchain.com/)
- **PDF Parsing Library**: [Unstructured](https://unstructured.io/)
- **Embedding Model**: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

---

### Flowchart

This ingestion process may be explained by the following (simplified) flowchart:

```mermaid
flowchart TD
    A["Start with a PDF file on the local system"] --> B["Use LangChain and Unstructured to load the document's text content"]
    B --> C["Use LangChain and Unstructured to split the full text into smaller, meaningful chunks"]
    C --> D["For each text chunk, use an embedding model to create a numerical vector"]
    D --> E["Use LangChain to connect to the local ChromaDB database"]
    E --> F["Store the original text chunks and their corresponding vectors in ChromaDB"]
    F --> G["End: Ingestion is complete. Data is stored securely and privately on the local machine."]
```

---

### Folder Explanation

- `1_ingestion`
    - The directory containing all logic related to data ingestion
- `reports`
    - The directory containing all reports to use
- `chromadb`
    - The directory containing the ChromaDB instance
- `testing`
    - The directory containing all pytest tests
- `docs`
    - The folder containing all documentation

---

## See the `docs` folder for flowcharts, prompts, schemas and more specific information.
