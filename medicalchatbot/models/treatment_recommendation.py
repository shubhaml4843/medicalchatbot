import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import cohere
import json
from datetime import datetime
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (COHERE_API_KEY, COHERE_MODEL, COHERE_PARAMS, DEVICE, MODELS, 
                   MEDICAL_CONFIG, DRUG_INTERACTIONS, CONTRAINDICATIONS)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not COHERE_API_KEY:
    logger.warning("Missing COHERE_API_KEY. AI-generated recommendations may be limited.")
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

def load_treatment_model():
    try:
        logger.info("Loading FLAN-T5 model...")
        flan_config = MODELS["flan_t5"]
        tokenizer = AutoTokenizer.from_pretrained(flan_config["name"])
        model = AutoModelForSeq2SeqLM.from_pretrained(flan_config["name"]).to(DEVICE)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading FLAN-T5: {e}")
        raise RuntimeError(f"Failed to load FLAN-T5 model: {e}")

FLAN_T5_MODEL, FLAN_T5_TOKENIZER = load_treatment_model()

def validate_patient_data(clinical_data: Dict) -> Tuple[str, str]:
    """Enhanced patient data validation"""
    logger.info(f"Processing clinical data: {len(str(clinical_data))} characters")

    if not clinical_data or not isinstance(clinical_data, dict):
        raise ValueError("Invalid clinical data: Must be a non-empty dictionary.")

    # Handle symptoms
    symptoms = clinical_data.get("symptoms", [])
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    symptoms_text = ", ".join(symptoms) if symptoms else "No symptoms provided"
    
    # Handle test results
    test_results = clinical_data.get("test_results", {})
    if isinstance(test_results, dict):
        test_results_text = "\n".join([f"{test}: {result}" for test, result in test_results.items()])
    else:
        test_results_text = str(test_results) if test_results else "No test results available"

    return symptoms_text, test_results_text

def check_drug_interactions(medications: List[str]) -> Dict[str, List[str]]:
    """Check for potential drug interactions"""
    interactions = {}
    for med1 in medications:
        med1_lower = med1.lower()
        if med1_lower in DRUG_INTERACTIONS:
            for med2 in medications:
                if med2.lower() in DRUG_INTERACTIONS[med1_lower]:
                    if med1 not in interactions:
                        interactions[med1] = []
                    interactions[med1].append(med2)
    return interactions

def calculate_treatment_confidence(clinical_data: Dict, diagnosis) -> float:
    """Calculate confidence score for treatment recommendations"""
    confidence = 0.5
    if clinical_data.get('symptoms'): confidence += 0.2
    if clinical_data.get('test_results'): confidence += 0.2
    if clinical_data.get('medical_history'): confidence += 0.1
    if 'complex' in str(diagnosis).lower(): confidence -= 0.2
    return min(max(confidence, 0.1), 1.0)

def generate_personalized_dosage(medication: str, age: int = 50, weight: float = 70) -> str:
    """Generate personalized medication dosage"""
    dosage_map = {
        'aspirin': f"{min(100, max(75, weight * 1.2))}mg daily",
        'metformin': f"{min(2000, max(500, weight * 15))}mg twice daily",
        'lisinopril': f"{min(40, max(5, age * 0.5))}mg daily"
    }
    
    for drug, dosage in dosage_map.items():
        if drug in medication.lower():
            return f"{medication} - {dosage}"
    return f"{medication} - Standard dosage"

def generate_treatment_plan(diagnosis, clinical_data: Dict) -> str:
    """Enhanced treatment plan generation with safety checks"""
    try:
        # Handle diagnosis as string or list
        if isinstance(diagnosis, list):
            diagnosis_str = ', '.join(str(d) for d in diagnosis if d)
        elif isinstance(diagnosis, dict):
            diagnosis_str = diagnosis.get('predicted_conditions', 'Unknown condition')
        else:
            diagnosis_str = str(diagnosis) if diagnosis else 'Unknown condition'
        
        symptoms, test_results = validate_patient_data(clinical_data)
        confidence = calculate_treatment_confidence(clinical_data, diagnosis_str)
        
        # Get patient info
        age = clinical_data.get('age', 50)
        weight = clinical_data.get('weight', 70)
        allergies = clinical_data.get('allergies', [])
        
        # Use Cohere for detailed treatment plan if available
        if co:
            cohere_prompt = f"""
You are a medical expert creating a comprehensive treatment plan.

Patient Information:
- Age: {age} years
- Weight: {weight} kg
- Allergies: {', '.join(allergies) if allergies else 'None'}
- Symptoms: {symptoms}
- Test Results: {test_results}
- Diagnosis: {diagnosis_str}

Create a detailed, structured treatment plan with:

1. PRIMARY MEDICATIONS:
   - Specific drug names with exact dosages
   - Administration frequency and timing
   - Duration of treatment

2. LIFESTYLE MODIFICATIONS:
   - Diet recommendations
   - Exercise guidelines
   - Activity restrictions
   - Sleep recommendations

3. MONITORING REQUIREMENTS:
   - Vital signs to monitor
   - Laboratory tests to repeat
   - Frequency of monitoring

4. FOLLOW-UP SCHEDULE:
   - Next appointment timing
   - Specialist referrals if needed
   - When to return if symptoms worsen

5. WARNING SIGNS:
   - Symptoms requiring immediate medical attention
   - When to contact healthcare provider
   - Emergency situations

Provide specific, actionable recommendations for each section."""
            
            try:
                cohere_response = co.chat(
                    model=COHERE_MODEL,
                    message=cohere_prompt,
                    temperature=0.3
                )
                detailed_plan = cohere_response.text.strip()
                if detailed_plan:
                    treatment_plan = detailed_plan
                else:
                    # Fallback to FLAN-T5
                    input_text = f"""
Patient: Age {age}, Weight {weight}kg, Allergies: {', '.join(allergies) if allergies else 'None'}
Symptoms: {symptoms}
Test Results: {test_results}
Diagnosis: {diagnosis_str}

Generate evidence-based treatment plan with:
1. Primary medications with dosages
2. Lifestyle modifications
3. Monitoring requirements
4. Follow-up schedule
5. Warning signs
Format as structured medical recommendations:"""
            except Exception as e:
                logger.error(f"Cohere API error: {e}")
                input_text = f"""
Patient: Age {age}, Weight {weight}kg, Allergies: {', '.join(allergies) if allergies else 'None'}
Symptoms: {symptoms}
Test Results: {test_results}
Diagnosis: {diagnosis_str}

Generate evidence-based treatment plan with:
1. Primary medications with dosages
2. Lifestyle modifications
3. Monitoring requirements
4. Follow-up schedule
5. Warning signs
Format as structured medical recommendations:"""
        else:
            input_text = f"""
Patient: Age {age}, Weight {weight}kg, Allergies: {', '.join(allergies) if allergies else 'None'}
Symptoms: {symptoms}
Test Results: {test_results}
Diagnosis: {diagnosis_str}

Generate evidence-based treatment plan with:
1. Primary medications with dosages
2. Lifestyle modifications
3. Monitoring requirements
4. Follow-up schedule
5. Warning signs
Format as structured medical recommendations:"""

        # Only use FLAN-T5 if Cohere didn't provide a response
        if 'treatment_plan' not in locals():
            flan_config = MODELS["flan_t5"]
            inputs = FLAN_T5_TOKENIZER(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding="longest",
                max_length=flan_config["max_length"]
            ).to(DEVICE)

            with torch.no_grad():
                output_tokens = FLAN_T5_MODEL.generate(
                    **inputs,
                    **flan_config["generation_params"]
                )

            treatment_plan = FLAN_T5_TOKENIZER.decode(output_tokens[0], skip_special_tokens=True).strip()
        
        # Extract medications and check interactions
        medications = re.findall(r'\b(?:prescribe|recommend|give)\s+([A-Za-z]+)', treatment_plan.lower())
        interactions = check_drug_interactions(medications)
        
        # Add confidence and safety info (no emojis for PDF compatibility)
        confidence_text = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
        
        enhanced_plan = f"TREATMENT PLAN (Confidence: {confidence_text} - {confidence:.1%})\n\n{treatment_plan}"
        
        if interactions:
            enhanced_plan += "\n\nWARNING - DRUG INTERACTIONS DETECTED:\n"
            for drug, interacts in interactions.items():
                enhanced_plan += f"- {drug} interacts with: {', '.join(interacts)}\n"
        
        # Add personalized dosages
        enhanced_plan += "\nPERSONALIZED DOSAGES:\n"
        for med in medications[:3]:
            enhanced_plan += f"- {generate_personalized_dosage(med, age, weight)}\n"
        
        return enhanced_plan

    except Exception as e:
        logger.error(f"Error generating treatment plan: {e}")
        return f"ERROR: Treatment plan generation failed: {str(e)}\nPlease consult healthcare professional."
