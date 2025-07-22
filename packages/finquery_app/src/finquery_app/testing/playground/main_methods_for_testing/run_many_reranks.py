from pprint import pprint

from langchain_core.runnables import RunnableConfig

from finquery_app.config import CHROMA_DB_PATH
from finquery_app.manager import get_langfuse_callback
from finquery_app.manager import get_vector_store, get_embeddings
from finquery_app.querying.query import execute_query_with_reranking_mlx
from finquery_app.querying.query import get_rag_test_questions

print("--- Initializing FinQuery Components ---")

questions = get_rag_test_questions()
embeddings = get_embeddings()
vector_store = get_vector_store("financial_documents", embeddings, str(CHROMA_DB_PATH))
langfuse_handler = get_langfuse_callback()
config = RunnableConfig(callbacks=[langfuse_handler], run_name="testing_chain_stuff")

print(f"--- Processing {len(questions)} questions ---")

for i, question in enumerate(questions):
    current_question_number = i + 1
    pprint("-------------------------------------------")
    pprint(f"Question {current_question_number}: {question}")
    answer_result = execute_query_with_reranking_mlx(question, vector_store, 10, 4, config)
    pprint(answer_result)
    pprint("-------------------------------------------")

"""
274/300 of the questions had the retrieval + reranking include the information needed to answer the question.
This was tested using Gemini 2.5 Pro as a judge w/ langfuse, running all test questions.
"""
