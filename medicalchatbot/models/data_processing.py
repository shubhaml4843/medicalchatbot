import os
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import csv
from docx import Document
import logging

# Logger Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Initialization for NER
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Embedding Model for FAISS Indexing (using smaller model)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Extract text from DOCX
def extract_docx_text(file_path):
    try:
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as e:
        raise RuntimeError(f"[ERROR] DOCX extraction failed: {e}")

# Extract text from JSON
def extract_json_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            return "\n".join(f"{k}: {v}" for k, v in json_data.items())
    except Exception as e:
        raise RuntimeError(f"[ERROR] JSON extraction failed: {e}")

# Extract text from CSV
def extract_csv_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            return "\n".join(", ".join(row) for row in reader)
    except Exception as e:
        raise RuntimeError(f"[ERROR] CSV extraction failed: {e}")

# Load clinical report (handles multiple formats)
def load_clinical_report(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = [doc.page_content for doc in loader.load()]
    elif file_extension in [".docx", ".doc"]:
        docs = [extract_docx_text(file_path)]
    elif file_extension == ".json":
        docs = [extract_json_text(file_path)]
    elif file_extension == ".csv":
        docs = [extract_csv_text(file_path)]
    elif file_extension == ".txt":
        loader = TextLoader(file_path)
        docs = [doc.page_content for doc in loader.load()]
    else:
        raise ValueError(f"[ERROR] Unsupported file format: {file_extension}")

    if not docs:
        raise ValueError("[ERROR] No content extracted from report.")
    
    # Chunk the document for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return [chunk for chunk in text_splitter.split_text("\n".join(docs)) if chunk.strip()]

# Merge NER entities
def merge_entities(entities):
    merged_entities = []
    current_entity = None

    for entity in entities:
        if entity["entity"].startswith("B-"):
            if current_entity:
                merged_entities.append(current_entity)
            current_entity = {"word": entity["word"], "label": entity["entity"][2:]}
        elif entity["entity"].startswith("I-") and current_entity:
            current_entity["word"] += f" {entity['word']}"
        else:
            if current_entity:
                merged_entities.append(current_entity)
                current_entity = None

    if current_entity:
        merged_entities.append(current_entity)

    return merged_entities

# Extract medical entities from chunks
def extract_medical_entities(chunks):
    extracted_entities = {"full_text": " ".join(chunks)}

    for chunk in chunks:
        with torch.no_grad():
            ner_results = ner_pipeline(chunk[:512])  # Limit input for Bio_ClinicalBERT

        for entity in merge_entities(ner_results):
            word, label = entity["word"].strip(), entity["label"].lower()

            if label not in extracted_entities:
                extracted_entities[label] = []
            extracted_entities[label].append(word)

    # Deduplicate extracted entities
    for key in extracted_entities:
        if isinstance(extracted_entities[key], list):
            extracted_entities[key] = list(set(extracted_entities[key]))

    return extracted_entities

# Create FAISS index
def create_faiss_index(text_chunks):
    if not text_chunks:
        raise ValueError("[ERROR] No valid text chunks for FAISS indexing.")
    return FAISS.from_texts(text_chunks, embedding_model)

# Main preprocessing function
def preprocess_report(file_path):
    try:
        report_chunks = load_clinical_report(file_path)
        
        if not report_chunks:
            raise ValueError("[ERROR] No valid content extracted from the report.")

        logger.info(f"[INFO] Loaded Report Chunks: {len(report_chunks)}")
        
        extracted_data = extract_medical_entities(report_chunks)
        
        if not any(extracted_data.values()):
            raise ValueError("[ERROR] No medical entities extracted from the report.")
        
        logger.info("[INFO] Extracted Medical Entities:", extracted_data)
        
        # Create FAISS index
        faiss_index = create_faiss_index(report_chunks)

        # Return clinical data and FAISS data
        clinical_data = extracted_data  # Medical entities and other extracted info
        faiss_data = faiss_index  # FAISS index for search
        return clinical_data, faiss_data

    except Exception as e:
        logger.error(f"[ERROR] Preprocessing failed: {e}")
        raise RuntimeError(f"[ERROR] Preprocessing failed: {e}")
