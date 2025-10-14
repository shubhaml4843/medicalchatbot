import os
import logging
import faiss
import numpy as np
import cohere
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from config import COHERE_API_KEY, COHERE_MODEL
import cohere
from .drug_interaction_checker import medical_ai
from .medical_ner import medical_ner
from .medical_knowledge_graph import medical_kg

# Load environment variables from current directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv()  # Also try current directory

# Logging
logger = get_logger(__name__)

# Initialize Cohere client
if not COHERE_API_KEY:
    logger.error("[ERROR] Missing COHERE_API_KEY in config.py")
    raise ValueError("[ERROR] Missing COHERE_API_KEY in config.py")

try:
    co = cohere.Client(COHERE_API_KEY)
    logger.info("[INFO] Cohere client initialized successfully.")
except Exception as e:
    logger.error(f"[ERROR] Failed to initialize Cohere client: {e}")
    raise

# Load Embedding Models (using smaller model)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS Setup
dimension = 384  # Dimension for MiniLM
index = faiss.IndexFlatL2(dimension)
report_texts = []


# Extract text from uploaded report
def extract_text_from_report(report_path):
    try:
        with open(report_path, "r", encoding="utf-8") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Report not found at {report_path}")


# Create or load FAISS index
def load_or_create_faiss(index_path, embedding_model, report_path=None):
    global report_texts
    faiss_files = [f"{index_path}/index.faiss", f"{index_path}/index.pkl"]
    if all(os.path.isfile(f) for f in faiss_files):
        logger.info("[INFO] Loading existing FAISS index...")
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

    if not report_path:
        raise FileNotFoundError("[ERROR] No FAISS index found and no report provided.")

    documents = extract_text_from_report(report_path)
    if not documents:
        raise ValueError("[ERROR] Report is empty. Cannot build FAISS index.")

    report_texts = documents
    logger.info("[INFO] Creating and saving new FAISS index...")
    faiss_index = FAISS.from_texts(documents, embedding_model)
    faiss_index.save_local(index_path)
    logger.info("[INFO] FAISS index created and saved successfully.")
    return faiss_index


# Get top matching info from report
def retrieve_relevant_info(question, faiss_index):
    try:
        if not faiss_index or not report_texts:
            return None
        query_vector = sentence_transformer.encode([question]).astype("float32")
        _, top_indices = faiss_index.index.search(query_vector, k=1                                                                                                         )
        if len(top_indices) > 0 and top_indices[0][0] != -1 and top_indices[0][0] < len(report_texts):
            return report_texts[top_indices[0][0]]
    except Exception as e:
        logger.error(f"Error retrieving info: {e}")
    return None
# Auto Classify and Respond Function (AI or Patient)
def classify_and_respond(question, faiss_index, clinical_data, chat_history=None):
    lower_q = question.lower().strip()

    # Short greetings
    if lower_q in ["hi", "hello", "hey"]:
        return "🔹 👤 Patient: 'Hi.'"

    # Check if asking about existing test results
    test_inquiry_keywords = [
        "any test", "test before", "previous test", "test result", "what test", 
        "test done", "lab result", "blood test", "x-ray", "scan", "ecg", "ekg"
    ]
    
    if any(kw in lower_q for kw in test_inquiry_keywords):
        # Return actual test results from clinical data
        test_results = clinical_data.get('test_results', {})
        if test_results and isinstance(test_results, dict):
            test_list = []
            for test, result in test_results.items():
                test_list.append(f"{test}: {result}")
            if test_list:
                return f"🔹 👤 Patient: 'Yes, I had these tests done: {', '.join(test_list)}'"
        return "🔹 👤 Patient: 'No, I haven't done any tests yet.'"

    # Keywords that should trigger AI (diagnosis, advice, treatment, medication)
    ai_trigger_keywords = [
        "diagnosis", "condition", "disease", "recommend", "suggest", "advise",
        "next step", "interpret", "what does it mean",
        "predict", "future", "plan", "what kind of diagnosis",
        "what do you recommend", "treatment", "medication", "drug", "medicine", "therapy", "prescribe",
        "i will give", "i will recommend", "i recommend", "i suggest", "i advise",
        "let me recommend", "let me suggest", "here is my recommendation", "my recommendation"
    ]

    # If any keyword is present, route to AI
    is_ai = any(kw in lower_q for kw in ai_trigger_keywords)

    # Prepare clinical context
    clinical_text = " ".join(f"{k}: {v}" for k, v in clinical_data.items())[:3000]
    history_text = "\n".join(chat_history[-3:]) if chat_history else ""

    # AI Mode - Enhanced with Advanced Features
    if is_ai:
        # Extract medical entities
        combined_text = f"{clinical_text} {question}"
        entities = medical_ner.extract_entities(combined_text)
        
        # Get clinical reasoning
        symptoms = entities.get('symptoms', [])
        conditions = entities.get('conditions', [])
        medications = entities.get('medications', [])
        
        clinical_reasoning = ""
        if symptoms:
            clinical_reasoning = medical_kg.generate_clinical_reasoning(symptoms, conditions)
        
        # Advanced AI Analysis
        drug_safety = ""
        severity_assessment = {}
        clinical_support = {}
        
        if len(medications) >= 2:
            drug_safety = medical_ai.get_safety_report(medications)
        
        if symptoms:
            severity_assessment = medical_ai.assess_symptom_severity(symptoms, clinical_data)
            clinical_support = medical_ai.clinical_decision_support(symptoms, {}, conditions)
        
        ai_response = co.chat(
            model=COHERE_MODEL,
            message=f"Doctor: {question}",
            preamble=f"""
You are an advanced AI medical assistant with enhanced analysis capabilities.
Use the clinical data and AI analysis to give medical advice.

Clinical Data:
{clinical_text}

AI Analysis:
- Symptoms: {', '.join(symptoms) if symptoms else 'None'}
- Conditions: {', '.join(conditions) if conditions else 'None'}
- Medications: {', '.join(medications) if medications else 'None'}

{clinical_reasoning}

{drug_safety}

Provide concise medical recommendations based on this enhanced analysis.""",
            max_tokens=200,
            temperature=0.3
        )
        response_text = ai_response.text.strip()
        
        # Add comprehensive AI analysis
        entity_summary = medical_ner.get_comprehensive_analysis(combined_text)
        
        # Format severity assessment
        severity_info = ""
        if severity_assessment:
            severity_info = f"\n🌡️ Severity: {severity_assessment.get('overall_severity', 'UNKNOWN')} - {severity_assessment.get('triage_level', 'CLINIC')}"
        
        # Format clinical decision support
        clinical_info = ""
        if clinical_support and clinical_support.get('primary_diagnosis'):
            clinical_info = f"\n🎯 Primary Diagnosis: {clinical_support['primary_diagnosis']}"
        
        analysis_note = f" [AI: {entity_summary['summary']}]{severity_info}{clinical_info}"
        
        return f"🔹 🤖 AI: '{response_text}'{analysis_note}"

    # Patient Mode
    else:
        patient_response = co.chat(
            model=COHERE_MODEL,
            message=f"Doctor: {question}",
            preamble=f"""
You are acting as a patient talking to your doctor.
Only answer as the patient using your own information.
Use short, human-like replies. Avoid long text. Be empathetic.
                                                                                                                                 
Patient Clinical Information:
{clinical_text}

Conversation History:
{history_text}""",
            max_tokens=60,
            temperature=0.4
        )
        response_text = patient_response.text.strip()
        return f"🔹 👤 Patient: '{response_text}'"

def chatbot_interaction(question, clinical_data, faiss_index, chat_history, mode="patient"):
    if not question.strip():
        return {
            "question": "👨‍⚕️ Doctor: (No input provided)",
            "answer": "🔹 🤖 AI: 'Please provide a question, doctor.'"
        }

    answer = classify_and_respond(question, faiss_index, clinical_data, chat_history)
    chat_history.append(f"Doctor: {question}\n{answer}")
    return {"question": f"👨‍⚕️ Doctor: '{question}'", "answer": answer}

