import os
import torch

# API Configuration
COHERE_API_KEY = "0BpDYSDqQjkX72aTfL2V3j2E4stGY2GEjFa4ew2K"  # Replace with your actual API key
COHERE_MODEL = "command-a-03-2025"  # Best model for medical AI

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Configuration
MODELS = {
    "biogpt": {
        "name": "microsoft/BioGPT-Large",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "cache_dir": None
    },
    "flan_t5": {
        "name": "google/flan-t5-large",
        "max_length": 1024,
        "generation_params": {
            "max_length": 800,
            "num_beams": 5,
            "temperature": 0.6,
            "repetition_penalty": 1.2,
            "early_stopping": True
        }
    },
    "embedding": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384
    }
}

# Medical AI Configuration
MEDICAL_CONFIG = {
    "confidence_threshold": 0.7,
    "max_medications": 5,
    "interaction_check": True,
    "personalization_level": "high",
    "max_context_length": 2048,
    "min_confidence_threshold": 0.6,
    "semantic_similarity_threshold": 0.75,
    "max_retrieved_docs": 5
}

# Cohere Generation Parameters
COHERE_PARAMS = {
    "temperature": 0.7,
    "max_tokens": None  # Use model default
}

# File Upload Configuration
UPLOAD_CONFIG = {
    "allowed_extensions": {"txt", "pdf", "docx", "doc"},
    "max_file_size": 16 * 1024 * 1024,  # 16MB
    "upload_folder": "uploads",
    "reports_folder": "reports"
}

# Medical Image Analysis Configuration
IMAGE_CONFIG = {
    "allowed_extensions": {"jpg", "jpeg", "png", "bmp", "tiff", "dcm"},
    "max_file_size": 50 * 1024 * 1024,  # 50MB for medical images
    "supported_types": ["xray", "mri", "ct", "ultrasound", "other"]
}

# FAISS Configuration
FAISS_CONFIG = {
    "index_path": "faiss_index",
    "dimension": 384,
    "k_results": 3
}

# Medical Knowledge Base
DRUG_INTERACTIONS = {
    "warfarin": ["aspirin", "ibuprofen", "vitamin k"],
    "metformin": ["alcohol", "contrast dye"],
    "insulin": ["alcohol", "beta-blockers"]
}

CONTRAINDICATIONS = {
    "pregnancy": ["ace_inhibitors", "warfarin", "tetracycline"],
    "kidney_disease": ["nsaids", "metformin"],
    "liver_disease": ["acetaminophen", "statins"]
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}