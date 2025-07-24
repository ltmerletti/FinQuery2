import pathlib
import threading
from typing import List

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from langchain_core.runnables import RunnableConfig

from finquery_app.chains.answer_chain import create_rag_chain
from finquery_app.config import SOURCE_DATA_DIR, SOURCE_PROCESSED_DATA_DIR, CHROMA_DB_PATH, COLLECTION_NAME, \
    LMSTUDIO_SMART_MODEL_NAME
from finquery_app.database.delete_collection import delete_collection_and_folder
from finquery_app.manager import get_vector_store, get_embeddings, get_langfuse_callback, \
    get_record_manager, get_llm, get_spacy_model, get_tiktoken_model
from finquery_app.querying.query import execute_query, get_rag_test_questions, execute_query_with_reranking
from finquery_app.ingestion.pipeline import run_ingestion_process

app = Flask(__name__)
CORS(app)

# --- Configuration ---
ALLOWED_EXTENSIONS = {'pdf'}

SOURCE_DATA_DIR.mkdir(exist_ok=True)
SOURCE_PROCESSED_DATA_DIR.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(SOURCE_DATA_DIR)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Variables ---
embeddings = get_embeddings()
vector_store = get_vector_store(COLLECTION_NAME, embeddings, CHROMA_DB_PATH)
record_manager = get_record_manager(COLLECTION_NAME)
smart_llm = get_llm(model_name=LMSTUDIO_SMART_MODEL_NAME)
retrieval_chain = create_rag_chain(vector_store, smart_llm)
handler = get_langfuse_callback()
llm = get_llm()
spacy_model = get_spacy_model()
tiktoken_model = get_tiktoken_model()

#  !!!!---------- API Endpoints ----------!!!!

@app.route("/api/health", methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "FinQuery API is running."}), 200

@app.route("/api/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = pathlib.Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(save_path)
        return jsonify({"message": f"File '{filename}' uploaded successfully."}), 201
    else:
        return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400

@app.route("/api/ingest", methods=['POST'])
def trigger_ingestion():
    try:
        ingestion_thread = threading.Thread(
            target=run_ingestion_process,
            args=(vector_store, record_manager, llm, spacy_model, tiktoken_model)
        )
        ingestion_thread.start()
        return jsonify({"status": "success", "message": "Ingestion process started in the background."}), 202
    except Exception as e:
        return jsonify({"error": f"Failed to start ingestion thread: {str(e)}"}), 500

@app.route("/api/query", methods=['POST'])
def query_documents():
    data = request.get_json()
    if not data or 'query_text' not in data:
        return jsonify({"error": "Request must include 'query_text'"}), 400

    query_text = data['query_text']
    num_to_fetch = data.get('num_to_fetch', 10)

    try:
        config = RunnableConfig(callbacks=[handler])
        results = execute_query(query_text, vector_store, num_to_fetch, config)
        formatted_results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        return jsonify({"query": query_text, "results": formatted_results}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred during query execution: {str(e)}"}), 500

@app.route("/api/query/rerank", methods=['POST'])
def query_documents_with_rerank():
    data = request.get_json()
    if not data or 'query_text' not in data:
        return jsonify({"error": "Request must include 'query_text'"}), 400

    query_text = data['query_text']
    num_to_fetch = data.get('num_to_fetch', 10)
    num_to_return = data.get('num_to_return', 4)

    try:
        config = RunnableConfig(callbacks=[handler])
        results = execute_query_with_reranking(query_text, vector_store, num_to_fetch, num_to_return, config)
        formatted_results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        return jsonify({"query": query_text, "results": formatted_results}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred during query execution: {str(e)}"}), 500

@app.route("/api/question", methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'query_text' not in data:
        return jsonify({"error": "Request must include 'query_text'"}), 400

    query_text = data['query_text']

    try:
        answer = retrieval_chain.invoke(query_text)
        return jsonify({"query": query_text, "answer": answer}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred during question answering: {str(e)}"}), 500


@app.route("/api/questions/batch", methods=['POST'])
def ask_questions_batch():
    data = request.get_json()
    if not data or 'query_texts' not in data or not isinstance(data['query_texts'], list):
        return jsonify({"error": "Request must include 'query_texts' as a list of strings"}), 400

    query_texts: List[str] = data['query_texts']

    valid_queries = [q for q in query_texts if q.strip()]
    if not valid_queries:
        return jsonify({"error": "The 'query_texts' list cannot be empty or contain only empty strings."}), 400

    try:
        config = RunnableConfig(callbacks=[handler])

        inputs_for_batch = valid_queries

        answers = retrieval_chain.batch(inputs_for_batch, config=config)

        results = [{"query": q, "answer": a} for q, a in zip(valid_queries, answers)]

        return jsonify({"results": results}), 200
    except Exception as e:
        import traceback
        app.logger.error(f"Error in batch question answering: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An error occurred during batch question answering: {str(e)}"}), 500



@app.route("/api/documents", methods=['GET'])
def get_documents_list():
    try:
        processed_files = [f.name for f in SOURCE_PROCESSED_DATA_DIR.iterdir() if f.is_file()]
        pending_files = [f.name for f in SOURCE_DATA_DIR.iterdir() if f.is_file() and f.name not in processed_files]
        return jsonify({"processed_documents": processed_files, "pending_ingestion": pending_files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status/db", methods=['GET'])
def get_db_status():
    try:
        count = vector_store._collection.count()
        return jsonify({"collection_name": COLLECTION_NAME, "total_chunks": count}), 200
    except Exception:
        return jsonify({"collection_name": COLLECTION_NAME, "total_chunks": 0, "status": "Collection may not exist yet."}), 200

@app.route("/api/testing/questions", methods=['GET'])
def get_test_questions():
    questions = get_rag_test_questions()
    return jsonify({"count": len(questions), "questions": questions}), 200

@app.route("/api/admin/collection", methods=['DELETE'])
def delete_db_collection():
    try:
        delete_collection_and_folder(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DB_PATH)
        )
        return jsonify({"status": "success", "message": f"Collection '{COLLECTION_NAME}' and its data have been deleted."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete collection: {str(e)}"}), 500

@app.route("/api/db/documents/content", methods=['GET'])
def get_all_document_content_from_db():
    try:
        return jsonify({"status": "success", "content":vector_store.get()}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete collection: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
