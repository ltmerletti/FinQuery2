import pathlib
import sys
import threading

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from querying.query import initialize_vector_store, execute_query, get_rag_test_questions
from ingestion.pipeline import run_ingestion_process
from ingestion.chromainit.delete_collection import delete_collection_and_folder
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler

app = Flask(__name__)
load_dotenv()
CORS(app)

# --- Configuration ---
project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
REPORTS_DIR = project_root / "reports"
PROCESSED_DIR = REPORTS_DIR / "added"
CHROMA_DB_DIR = project_root / "chromadb"
COLLECTION_NAME = "financial_documents"
ALLOWED_EXTENSIONS = {'pdf'}

REPORTS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(REPORTS_DIR)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        ingestion_thread = threading.Thread(target=run_ingestion_process)
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
        handler = CallbackHandler()
        config = RunnableConfig(callbacks=[handler])
        vector_store = initialize_vector_store(
            persist_directory=str(CHROMA_DB_DIR),
            collection_name=COLLECTION_NAME
        )
        results = execute_query(query_text, vector_store, num_to_fetch, config)
        formatted_results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        return jsonify({"query": query_text, "results": formatted_results}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred during query execution: {str(e)}"}), 500

@app.route("/api/documents", methods=['GET'])
def get_documents_list():
    try:
        processed_files = [f.name for f in PROCESSED_DIR.iterdir() if f.is_file()]
        pending_files = [f.name for f in REPORTS_DIR.iterdir() if f.is_file() and f.name not in processed_files]
        return jsonify({"processed_documents": processed_files, "pending_ingestion": pending_files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status/db", methods=['GET'])
def get_db_status():
    try:
        vector_store = initialize_vector_store(
            persist_directory=str(CHROMA_DB_DIR),
            collection_name=COLLECTION_NAME
        )
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
            persist_directory=str(CHROMA_DB_DIR)
        )
        return jsonify({"status": "success", "message": f"Collection '{COLLECTION_NAME}' and its data have been deleted."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete collection: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
