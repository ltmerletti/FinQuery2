import json
import threading
import pathlib

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Local Application Imports
from finquery_app.config import (
    LMSTUDIO_SMART_MODEL_NAME, LMSTUDIO_FAST_LLM_MODEL_NAME, DB_NAME, DB_USER,
    DB_PASSWORD, DB_HOST, DB_PORT, COLLECTION_NAME, CHROMA_DB_PATH, SOURCE_DATA_DIR
)
from finquery_app.ingestion.pipeline import run_ingestion_process
from finquery_app.manager import (
    get_vector_store, get_embeddings, get_record_manager, get_llm,
    get_spacy_model, get_tiktoken_model
)
from finquery_app.chains.conversational_refiner import create_conversational_refiner_chain
from finquery_app.chains.answer_chain import create_rag_chain
from finquery_parser.types import PostgresDBConnector

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
ALLOWED_EXTENSIONS = {'pdf'}
SOURCE_DATA_DIR.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(SOURCE_DATA_DIR)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


print("--- Initializing FinQuery API Components ---")
embeddings = get_embeddings()
vector_store = get_vector_store(COLLECTION_NAME, embeddings, CHROMA_DB_PATH)
record_manager = get_record_manager(COLLECTION_NAME)
smart_llm = get_llm(model_name=LMSTUDIO_SMART_MODEL_NAME)
fast_llm = get_llm(model_name=LMSTUDIO_FAST_LLM_MODEL_NAME)
spacy_model = get_spacy_model()
tiktoken_model = get_tiktoken_model()
db_connector = PostgresDBConnector(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
refiner_chain = create_conversational_refiner_chain(smart_llm, db_connector)
print("--- All Components Initialized ---")


#  !!!!---------- API Endpoints ----------!!!!

@app.route("/api/health", methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "FinQuery API is running."}), 200


@app.route("/api/chat", methods=['POST'])
def chat_handler():
    """
    Handles the entire stateful, conversational interaction for question-answering.
    This single endpoint manages both query refinement and final answer retrieval.
    """
    data = request.get_json()
    if not data or 'message' not in data or 'session_id' not in data:
        return jsonify({"error": "Request must include 'message' and 'session_id'"}), 400

    user_message = data['message']
    session_id = data['session_id']

    try:
        response = refiner_chain.invoke(
            {"question": user_message},
            config={"configurable": {"session_id": session_id}},
        )
        action = response.get("action")

        if action == "ask":
            return jsonify({"type": "ask", "message": response.get("question")}), 200

        elif action == "filter":
            query_data = response.get("data", {})
            search_query = query_data.get("search_query", user_message)
            metadata_filter = query_data.get("metadata_filter", {})

            rag_chain = create_rag_chain(vector_store, smart_llm, metadata_filter)

            def stream_answer():
                yield json.dumps({"type": "answer_start"}) + "\n"
                for chunk in rag_chain.stream(search_query):
                    yield json.dumps({"type": "answer_chunk", "content": chunk}) + "\n"
                yield json.dumps({"type": "end_of_stream"}) + "\n"

            return Response(stream_answer(), mimetype='application/json')

        else:
            return jsonify({"error": "Unknown action from refinement chain."}), 500

    except Exception as e:
        app.logger.error(f"Error in chat handler: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/api/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(pathlib.Path(app.config['UPLOAD_FOLDER']) / filename)
        return jsonify({"message": f"File '{filename}' uploaded."}), 201
    return jsonify({"error": "Invalid file type"}), 400


@app.route("/api/ingest", methods=['POST'])
def trigger_ingestion():
    """Triggers the data ingestion process in a background thread."""
    try:
        ingestion_thread = threading.Thread(
            target=run_ingestion_process,
            args=(vector_store, record_manager, fast_llm, smart_llm, spacy_model, tiktoken_model, db_connector)
        )
        ingestion_thread.start()
        return jsonify({"status": "success", "message": "Ingestion started."}), 202
    except Exception as e:
        return jsonify({"error": f"Failed to start ingestion: {str(e)}"}), 500


@app.route("/api/db/status", methods=['GET'])
def get_db_status():
    try:
        count = vector_store._collection.count()
        return jsonify({"collection_name": COLLECTION_NAME, "total_chunks": count}), 200
    except Exception:
        return jsonify({"collection_name": COLLECTION_NAME, "total_chunks": 0, "status": "Collection not found."}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5001)
