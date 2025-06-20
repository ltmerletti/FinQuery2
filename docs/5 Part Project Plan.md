1. Objectives

- To provide users with a way to readily retrieve financial information from a database and perform operations using
  natural language
- To simplify the process of storage for easy and automatic retrieval of information
- To also have a frontend for easy use

2. Process (in English)

- The user will upload a variety of financial form PDFs to the service
- The service will extract key information and store it in a structured format for retrieval
- The user will ask a question (such as “what was the percent change in revenue from quarter 1 to quarter 3 of 2022?”)
- The system will provide a direct answer to the user stating their information plainly. It will have a trace of where
  specifically it retrieved the data and what operations were performed.

3. Data sources

- In production, primarily PDFs of financial data from clients/employees like yearly reports, tax returns, etc.
- For testing purposes the widely available company report PDFs will be taken from the internet and/or sites like Yahoo
  Finance

4. Operations to perform (like database, data fetching etc.)

- PDF Parsing/Size reduction
- Using some library to reduce the size of the PDF files given and then parse it
- SQL-Based Structured Database
- Storing the extracted/parsed data from the PDF in the database in a highly-structured way
- Database fetching
- Retrieving specific/limited data from the database using SQL queries
- Translation of natural language to database queries and calculations
- Using an LLM agent to parse natural language from the user into a database query
- sing a second LLM agent to create calculation code based on the user request

5. External integration

- Hosting environment
    - Potentially Azure or Google cloud, unless there are other servers to host this on
    - This would include the database hosting as well
- Google Gemini API
    - For it’s long context length + good performance on high-context retrieval benchmarks
