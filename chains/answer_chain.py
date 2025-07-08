import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from querying.query import initialize_vector_store

def format_docs(docs: list) -> str:
    formatted_snippets = []

    for doc in docs:
        try:
            content_start_index = doc.page_content.index("[CONTENT]")
            content = doc.page_content[content_start_index:].replace("[CONTENT]", "").strip()
        except ValueError:
            content = doc.page_content.strip()

        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "N/A")

        snippet = (f"Source: {source}, Page: {page}\n"
                   f"Content:\n"
                   f"{content}")
        formatted_snippets.append(snippet)

    return "\n\n---\n\n".join(formatted_snippets)


def create_answer_chain(vector_store):
    load_dotenv()

    template = """You are an expert financial analyst AI. Your task is to provide a precise answer to the user's question based *only* on the context provided from financial documents.

Follow these steps rigorously:
1.  Carefully read the user's question to understand exactly what information is being asked for.
2.  Review each of the context snippets below. Each snippet is from a specific source document and page.
3.  Identify the single snippet that most directly and accurately answers the user's question. Ignore snippets that are only tangentially related or do not contain the specific data point requested.
4.  If no snippet contains the answer, respond with: "I could not find the answer in the provided documents."
5.  If a relevant snippet is found, construct your answer by directly extracting the information. State the fact or figure clearly and concisely.
6.  After providing the answer, you MUST cite your source in the format: "(Source: [filename], Page: [page number])".

Do not add any preamble, conversational text, or information that is not from the provided context.

---
CONTEXT SNIPPETS:
{context}
---
USER QUESTION:
{question}
---
PRECISE ANSWER:"""

    retriever = vector_store.as_retriever(search_kwargs={'k': 4})

    gather_inputs = {"context": retriever | format_docs, "question": RunnablePassthrough()}

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0, model=os.getenv("LMSTUDIO_MODEL_NAME"), base_url=os.getenv("LMSTUDIO_BASE_URL"),
        api_key=os.getenv("LMSTUDIO_API_KEY"))

    output_parser = StrOutputParser()

    rag_chain = gather_inputs | prompt | llm | output_parser

    return rag_chain


def get_llm_output(rag_chain, question: str) -> str:
    return rag_chain.invoke(question)


if __name__ == "__main__":
    test_question = "What were Apple's total net sales in 2023?"
    vector_store = initialize_vector_store()
    rag_chain = create_answer_chain(vector_store)
    output = get_llm_output(rag_chain, test_question)
    print(output)