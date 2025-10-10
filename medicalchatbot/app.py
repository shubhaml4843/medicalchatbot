import os
import json
import torch
import logging
import time
from flask import Flask, render_template, request, jsonify, send_file
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import get_logger
from werkzeug.utils import secure_filename
from config import FAISS_CONFIG, UPLOAD_CONFIG
from models import (
    preprocess_report,
    chatbot_interaction,
    generate_pdf_report,
    diagnose,
    generate_test_recommendations,
    generate_treatment_plan,
    generate_followup_plan,
    extract_report_data,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# App initialization
app = Flask(__name__)

# Logging setup
logger = get_logger(__name__)

# Directories from config
UPLOAD_FOLDER = UPLOAD_CONFIG["upload_folder"]
FAISS_INDEX_PATH = FAISS_CONFIG["index_path"]
REPORTS_FOLDER = UPLOAD_CONFIG["reports_folder"]
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Session-based storage instead of global variables
from flask import session
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# File validation from config
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in UPLOAD_CONFIG["allowed_extensions"]

def validate_file_size(file):
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    return size <= UPLOAD_CONFIG["max_file_size"]

# Initialize embedding model
def initialize_embedding_model():
    try:
        logger.info("Loading Hugging Face embedding model...")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        logger.exception("Error loading embedding model: %s", e)
        raise

embedding_model = initialize_embedding_model()

# FAISS index helpers
def create_faiss_index(text_chunks):
    try:
        if not text_chunks or len(text_chunks) == 0:
            logger.warning("No valid text chunks provided. Creating with dummy data.")
            text_chunks = ["Medical knowledge base placeholder"]
        
        logger.info("Creating FAISS index...")
        index = FAISS.from_texts(text_chunks, embedding_model)
        index.save_local(FAISS_INDEX_PATH)
        return index
    except Exception as e:
        logger.exception("Error creating FAISS index: %s", e)
        # Return empty index as fallback
        return FAISS.from_texts(["Medical knowledge base placeholder"], embedding_model)

def load_faiss_index():
    try:
        logger.info("Loading FAISS index from local storage...")
        return FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.warning(f"No FAISS index found: {e}. Creating empty index.")
        # Create empty index with dummy text to avoid IndexError
        dummy_texts = ["Medical knowledge base placeholder"]
        return FAISS.from_texts(dummy_texts, embedding_model)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_report", methods=["POST"])
def upload_report():
    try:
        # Validate file presence
        if 'clinical_report' not in request.files:
            return jsonify({"success": False, "error": "No file part in request."}), 400
        
        file = request.files['clinical_report']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected."}), 400
        
        # Security validations
        if not file or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type. Only txt, pdf, docx, doc allowed."}), 400
        
        if not validate_file_size(file):
            return jsonify({"success": False, "error": "File too large. Maximum 16MB allowed."}), 400

        # Secure file handling
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"success": False, "error": "Invalid filename."}), 400
        
        # Add timestamp to prevent conflicts
        import time
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Validate file path is within upload directory
        if not os.path.abspath(filepath).startswith(os.path.abspath(UPLOAD_FOLDER)):
            return jsonify({"success": False, "error": "Invalid file path."}), 400
        
        file.save(filepath)
        logger.info(f"File uploaded successfully: {filename}")

        # Process report
        clinical_data, faiss_data = preprocess_report(filepath)
        faiss_index = faiss_data or create_faiss_index(clinical_data.get("text_chunks", []))
        
        # Store in session instead of global
        session['clinical_data'] = clinical_data
        session['has_data'] = True
        
        # Clean up uploaded file after processing
        try:
            os.remove(filepath)
        except OSError:
            logger.warning(f"Could not remove temporary file: {filepath}")

        return jsonify({"success": True, "message": "Report processed successfully."})
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"success": False, "error": "Invalid input data."}), 400
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return jsonify({"success": False, "error": "File processing failed."}), 500
    except Exception as e:
        logger.exception("Unexpected error processing report")
        return jsonify({"success": False, "error": "Internal server error."}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        # Authorization check
        if not session.get('has_data'):
            return jsonify({"success": False, "error": "Upload a clinical report first."}), 401
        
        clinical_data = session.get('clinical_data')
        if not clinical_data:
            return jsonify({"success": False, "error": "No clinical data available."}), 400

        # Input validation
        if not request.json:
            return jsonify({"success": False, "error": "Invalid request format."}), 400
            
        question = request.json.get("question", "").strip()
        if not question or len(question) > 1000:  # Limit question length
            return jsonify({"success": False, "error": "Invalid question format or too long."}), 400

        # Load FAISS index
        faiss_index = load_faiss_index()
        if faiss_index is None:
            return jsonify({"success": False, "error": "Search index not available."}), 500
        
        # Get or initialize chat history from session
        chat_history = session.get('chat_history', [])
        
        # Limit chat history size
        if len(chat_history) > 50:
            chat_history = chat_history[-25:]  # Keep last 25 entries

        response = chatbot_interaction(question, clinical_data, faiss_index, chat_history)
        
        # Update session chat history
        session['chat_history'] = chat_history

        return jsonify({"success": True, "response": response.get("answer", "No answer found.")})

    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        return jsonify({"success": False, "error": "Invalid input."}), 400
    except Exception as e:
        logger.exception("Error processing question")
        return jsonify({"success": False, "error": "Internal server error."}), 500

@app.route("/generate_report", methods=["POST"])
def generate_report():
    try:
        logger.info("Report generation requested")

        # Authorization check
        if not session.get('has_data'):
            return jsonify({"success": False, "error": "Upload a clinical report first."}), 401
        
        clinical_data = session.get('clinical_data')
        if not clinical_data:
            return jsonify({"success": False, "error": "No clinical data available."}), 400

        # Extract structured report data with error handling
        try:
            report_data = extract_report_data(clinical_data.get("full_text", ""))
            if not report_data:
                raise ValueError("No data extracted from report")
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return jsonify({"success": False, "error": "Failed to process clinical data."}), 500

        # Validate required fields
        required_fields = ["name", "age", "gender", "symptoms", "test_results", "vital_signs"]
        missing_fields = [f for f in required_fields if f not in report_data or not report_data[f]]

        if missing_fields:
            logger.warning(f"Missing fields: {missing_fields}")
            # Provide default values for missing fields
            for field in missing_fields:
                if field in ["symptoms", "test_results", "vital_signs"]:
                    report_data[field] = report_data.get(field, {})
                else:
                    report_data[field] = "Not specified"

        # Generate PDF report with comprehensive error handling
        try:
            report_path = generate_pdf_report(report_data)
            if not report_path or not os.path.isfile(report_path):
                raise FileNotFoundError("Generated report file not found")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return jsonify({"success": False, "error": "Report generation failed."}), 500

        # Validate file path security
        if not os.path.abspath(report_path).startswith(os.path.abspath("reports")):
            logger.error(f"Invalid report path: {report_path}")
            return jsonify({"success": False, "error": "Invalid report path."}), 500

        logger.info(f"Report generated successfully: {os.path.basename(report_path)}")
        return send_file(report_path, as_attachment=True, download_name=f"medical_report_{int(time.time())}.pdf")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return jsonify({"success": False, "error": "Report file not found."}), 404
    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        return jsonify({"success": False, "error": "File access denied."}), 403
    except Exception as e:
        logger.exception("Unexpected error generating report")
        return jsonify({"success": False, "error": "Internal server error."}), 500
# Start app
if __name__ == "__main__":
    # Security: Disable debug in production
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 8040))
    
    if debug_mode:
        logger.warning("Running in debug mode - not suitable for production")
    
    app.run(debug=debug_mode, port=port, host='127.0.0.1')
