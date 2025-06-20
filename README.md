```
▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖ ▗▖ ▗▖▗▄▄▄▖▗▄▄▖▗▖  ▗▖
▐▌     █  ▐▛▚▖▐▌▐▌ ▐▌ ▐▌ ▐▌▐▌   ▐▌ ▐▌▝▚▞▘ 
▐▛▀▀▘  █  ▐▌ ▝▜▌▐▌ ▐▌ ▐▌ ▐▌▐▛▀▀▘▐▛▀▚▖ ▐▌  
▐▌   ▗▄█▄▖▐▌  ▐▌▐▙▄▟▙▖▝▚▄▞▘▐▙▄▄▖▐▌ ▐▌ ▐▌                                          
```

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![React 19.1.0](https://img.shields.io/badge/React-19.1.0-blue?logo=react)
![MySQL 9.3.0](https://img.shields.io/badge/MySQL-9.3.0-blue?logo=mysql)

This app parses PDF documents into a secure SQL database using an AI parser. Users ask questions, and the system
understands the question, generates a SQL query, and calculates the answer. The result is returned with a trace of data
and calculations for validation.

**Key Steps:**

1. Upload PDF → Parse with AI → Store in encrypted DB
2. User asks question → AI understands and generates math query
3. SQL is parsed, run, and math process is subsequently executed securely
4. Final answer with trace is returned to user

---

**Tech Stack**

- Primary Language, Backend: Python
- Test Framework, Backend: pytest
- API Framework, Backend: FastAPI
- Database: MySQL
- Frontend: React JS

---
**Flowchart**

This process may be explained by the following (simplified) flowchart:

![Flow Overview](flow_overview.png "Flow Overview")

---

**Folder Explanation**

- `document_ingestion_agent`
    - Is the AI agent responsible for taking in PDFs and outputting formatted JSON (which will be put into the database)
- `math_interpreter_agent`
    - Is the AI agent responsible for converting userinput into database queries and math equations
- `sql_repair_agent`
    - Is the AI agent used to repair incorrect SQL queries from the `math_interpreter_agent`
- `conversation_agent`
    - Is the AI agent used to talk with the user
- `math_and_sql_execution_service`
    - Executes the SQL queries given by the `math_interpreter_agent`/`sql_repair_agent` and runs the math equations
- `storage_service`
    - Stores the output of the `document_ingestion_agent` in the database and converts SELECT query responses to JSON
- `finquery-frontend`
    - The frontend of the program written in React
- `testing`
    - The directory holding all pytest tests
- `docs`
    - The folder containing all documentation

---

**See the `docs` folder for flowcharts, prompts, schemas and more specific information.**
