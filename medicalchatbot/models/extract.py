import re
import json
import logging
from transformers import pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

# Logger Configuration
logger = get_logger(__name__)

# Load Clinical NER Model (Lightweight alternative)
try:
    # Use smaller clinical model instead of BioGPT-Large
    biogpt_ner = pipeline("ner", model="emilyalsentzer/Bio_ClinicalBERT")
    logger.info("Clinical NER model loaded successfully.")
except Exception as e:
    logger.warning("Clinical NER model not available, using fallback: %s", e)
    biogpt_ner = None


def is_valid_symptom(symptom):
    """Advanced symptom validation with medical terminology."""
    symptom = symptom.lower().strip()
    
    # Skip numbers, dates, measurements
    if symptom.isdigit() or re.match(r"^\d+\s*(minutes|hours|days|weeks|months|years|mg|ml|°c|°f|bpm|mmhg)?$", symptom):
        return False
    
    # Medical symptom keywords (positive indicators)
    medical_symptoms = {
        "pain", "ache", "fever", "cough", "nausea", "vomiting", "diarrhea", "constipation",
        "headache", "dizziness", "fatigue", "weakness", "shortness", "breathing", "chest",
        "abdominal", "swelling", "rash", "itching", "burning", "tingling", "numbness"
    }
    
    # Invalid terms (negative indicators)
    invalid_terms = {
        "reference range", "height", "ray", "drinks", "segment depression",
        "lowering agent", "lead inversion", "normal", "abnormal", "test", "result",
        "report", "patient", "doctor", "hospital", "clinic", "date", "time"
    }
    
    # Check if contains medical symptom keywords
    has_medical_term = any(term in symptom for term in medical_symptoms)
    has_invalid_term = any(term in symptom for term in invalid_terms)
    
    return has_medical_term and not has_invalid_term and len(symptom) > 2


def extract_report_data(report_content):
    """Extract structured medical information using BioGPT and regex."""
    logger.info(f"Received report content:\n{report_content}")

    if not report_content or not isinstance(report_content, str) or len(report_content.strip()) == 0:
        logger.error("No valid report content provided.")
        return {"error": "No report content provided."}

    extracted_data = {
        "full_report": report_content.strip(),
        "name": "Not Found",
        "age": "Not Found",
        "gender": "Not Found",
        "medical_history": "Not Found",
        "symptoms": [],
        "vital_signs": {},
        "test_results": {},
        "medications": [],
        "preliminary_diagnosis": "Not Found"
    }

    # Step 1: Extract Name, Age, Gender
    name_match = re.search(r"Name\s*:\s*([A-Za-z\s]+)", report_content)
    age_match = re.search(r"Age\s*:\s*(\d+)", report_content)
    gender_match = re.search(r"Gender\s*:\s*(Male|Female|Other)", report_content, re.IGNORECASE)

    if name_match:
        extracted_data["name"] = name_match.group(1).strip()
    if age_match:
        extracted_data["age"] = age_match.group(1).strip()
    if gender_match:
        extracted_data["gender"] = gender_match.group(1).capitalize()
    
    # Step 2: Extract Medical History
    history_match = re.search(
        r"Medical History\s*:?\s*(.*?)(?:\n(?:Medications|Lifestyle Factors|Preliminary Diagnosis|Treatment Plan|Diagnosis|Tests Recommended|Symptoms|Vitals|Physical Examination)\s*:|\n_{5,}|\Z)",
        report_content,
        re.IGNORECASE | re.DOTALL
    )
    if history_match:
        history_raw = history_match.group(1).strip()
        if history_raw:
            extracted_data["medical_history"] = history_raw


    # Step 2: Extract Entities using Clinical NER (if available)
    if biogpt_ner:
        try:
            # Process in smaller chunks to avoid memory issues
            chunk_size = 512
            for i in range(0, len(report_content), chunk_size):
                chunk = report_content[i:i+chunk_size]
                entities = biogpt_ner(chunk)
                
                for entity in entities:
                    entity_type = entity.get('entity_group', entity.get('label', None))
                    word = entity.get('word', "").strip()

                    if entity_type and word:
                        if "SYMPTOM" in str(entity_type).upper() or "DISEASE" in str(entity_type).upper():
                            if is_valid_symptom(word):
                                extracted_data["symptoms"].append(word)
                        elif "MEDICATION" in str(entity_type).upper():
                            extracted_data["medications"].append(word)
        except Exception as e:
            logger.warning("Error processing clinical entities: %s", e)

    # Step 3: Extract Preliminary Diagnosis via Regex (improved logic)
    diagnosis_patterns = [
        r"Preliminary Diagnosis\s*[:\-]\s*([^\n]+)",
        r"Provisional Diagnosis\s*[:\-]\s*([^\n]+)",
        r"Clinical Impression\s*[:\-]\s*([^\n]+)",
        r"Diagnosis\s*[:\-]\s*([^\n]+)",
        r"Impression\s*[:\-]\s*([^\n]+)"
    ]
    for pattern in diagnosis_patterns:
        match = re.search(pattern, report_content, re.IGNORECASE)
        if match:
            diagnosis = match.group(1).strip()
            # Clean up diagnosis text
            diagnosis = re.sub(r'[{}\[\]"\']', '', diagnosis)
            if diagnosis and diagnosis.lower() not in ["", "none", "not found", "unknown"]:
                extracted_data["preliminary_diagnosis"] = diagnosis[:200]  # Limit length
                break

    # Step 4: Extract Symptoms via Regex fallback
    symptom_patterns = [
        r"Symptoms\s*:\s*(.*?)(?:\n[A-Z]|\n$)",
        r"(-\s*[\w\s]+)",
        r"([\w\s]+):\s*\(Duration"
    ]
    extracted_symptoms = []
    for pattern in symptom_patterns:
        matches = re.findall(pattern, report_content, re.DOTALL)
        for match in matches:
            symptoms_list = match.strip().split("\n")
            extracted_symptoms.extend([s.strip("- ").strip() for s in symptoms_list])
    valid_symptoms = set(filter(is_valid_symptom, extracted_symptoms))
    extracted_data["symptoms"].extend(valid_symptoms)

    # Step 5: Extract Vital Signs (with fallback patterns)
    vital_signs_patterns = {
        "heart_rate": [r"Heart Rate\s*:\s*(\d+)\s*bpm", r"HR\s*:\s*(\d+)", r"Pulse\s*:\s*(\d+)"],
        "blood_pressure": [r"Blood Pressure\s*:\s*(\d+/\d+)", r"BP\s*:\s*(\d+/\d+)"],
        "temperature": [r"Temperature\s*:\s*([\d.]+)\s*°?C?", r"Temp\s*:\s*([\d.]+)"],
        "respiratory_rate": [r"Respiratory Rate\s*:\s*(\d+)\s*bpm", r"RR\s*:\s*(\d+)"],
        "oxygen_saturation": [r"Oxygen Saturation\s*:\s*(\d+)\s*%", r"O2 Sat\s*:\s*(\d+)\s*%"]
    }
    
    for key, patterns in vital_signs_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, report_content, re.IGNORECASE)
            if match:
                extracted_data["vital_signs"][key] = match.group(1).strip()
                break
    
    # Add default vital signs if none found
    if not extracted_data["vital_signs"]:
        extracted_data["vital_signs"] = {
            "heart_rate": "Normal",
            "blood_pressure": "Normal",
            "temperature": "Normal"
        }

    # Step 6: Enhanced Test Results Extraction
    test_keywords = {
        "cholesterol", "ldl", "hdl", "triglycerides", "hba1c", "glucose", "hemoglobin",
        "platelet", "wbc", "rbc", "creatinine", "bilirubin", "urea", "sodium", "potassium",
        "calcium", "phosphorus", "albumin", "protein", "ast", "alt", "alkaline", "phosphatase",
        "troponin", "bnp", "d-dimer", "pt", "ptt", "inr", "esr", "crp", "tsh", "t3", "t4"
    }
    
    # Multiple patterns for test results
    test_patterns = [
        r"([A-Za-z0-9 _-]+)\s*:\s*([\d.]+\s*[a-zA-Z/]*|Positive|Negative|Normal|Abnormal|High|Low)",
        r"([A-Za-z0-9 _-]+)\s*-\s*([\d.]+\s*[a-zA-Z/]*|Positive|Negative|Normal|Abnormal)",
        r"([A-Za-z0-9 _-]+)\s*=\s*([\d.]+\s*[a-zA-Z/]*|Positive|Negative|Normal|Abnormal)"
    ]
    
    for pattern in test_patterns:
        test_results = re.findall(pattern, report_content, re.IGNORECASE)
        for test, value in test_results:
            test_lower = test.lower().strip()
            if any(keyword in test_lower for keyword in test_keywords) and len(test.strip()) > 2:
                extracted_data["test_results"][test.strip()] = value.strip()

    # Step 7: Extract Medications (improved)
    medication_patterns = [
        r"Medication[s]?\s*:\s*([^\n]+)",
        r"Prescribed\s*Medication[s]?\s*:\s*([^\n]+)",
        r"Treatment\s*:\s*([^\n]+)"
    ]
    
    for pattern in medication_patterns:
        medication_match = re.search(pattern, report_content, re.IGNORECASE)
        if medication_match:
            med_text = medication_match.group(1).strip()
            # Clean and split medications
            meds = [med.strip() for med in re.split(r'[,;\n]', med_text) if med.strip()]
            # Filter out non-medication text
            clean_meds = [med for med in meds if len(med) < 100 and not any(x in med.lower() for x in ['lifestyle', 'diagnosis', 'factors'])]
            extracted_data["medications"].extend(clean_meds)
            break

    # Step 8: Remove Duplicates
    extracted_data["symptoms"] = list(set(extracted_data["symptoms"]))
    extracted_data["medications"] = list(set(extracted_data["medications"]))

    # Step 9: Final Checks for Missing Fields
    required_fields = ["name", "age", "gender", "symptoms", "test_results", "medical_history", "vital_signs", "medications"]
    for field in required_fields:
        if not extracted_data.get(field) or extracted_data[field] == "Not Found":
            extracted_data[field] = "Unknown" if isinstance(extracted_data[field], str) else []

    logger.info(f"Final Extracted Data:\n{json.dumps(extracted_data, indent=4)}")
    return extracted_data
