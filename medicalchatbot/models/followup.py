import torch
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import cohere
from models.diagnosis import diagnose
from models.extract import extract_report_data
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import COHERE_API_KEY, COHERE_MODEL, DEVICE, MODELS, COHERE_PARAMS

# Configure logging
logger = get_logger(__name__)

# Load BioGPT Model with memory optimization
try:
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    biogpt_config = MODELS["biogpt"]
    
    TOKENIZER = AutoTokenizer.from_pretrained(biogpt_config["name"], cache_dir=biogpt_config["cache_dir"])
    MODEL = AutoModelForCausalLM.from_pretrained(
        biogpt_config["name"],
        torch_dtype=biogpt_config["torch_dtype"],
        low_cpu_mem_usage=biogpt_config["low_cpu_mem_usage"],
        cache_dir=biogpt_config["cache_dir"]
    )
    MODEL = MODEL.to(DEVICE)
    logger.info(f"BioGPT Model Loaded on: {DEVICE}")
except Exception as e:
    logging.error(f"Failed to load BioGPT: {e}")
    MODEL = None
    TOKENIZER = None

# Load Cohere API
if not COHERE_API_KEY:
    logger.warning("Missing COHERE_API_KEY. Cohere features will be disabled.")
    co = None
else:
    co = cohere.Client(COHERE_API_KEY)

# Generate Follow-Up Plan
def generate_followup_plan(symptoms, diagnosis, test_results):
    """
    Generates a follow-up plan based on clinical data, predicted diagnosis, and AI-generated recommendations.
    """
    if not symptoms:
        raise ValueError("Symptoms data is required.")
    
    # Handle diagnosis parameter (can be string, list, or dict)
    if isinstance(diagnosis, list):
        diagnosis_str = ', '.join(str(d) for d in diagnosis if d)
    elif isinstance(diagnosis, dict):
        diagnosis_str = diagnosis.get('predicted_conditions', diagnosis.get('predicted_diagnosis', 'General condition'))
    else:
        diagnosis_str = str(diagnosis) if diagnosis else 'General condition'
    
    # Step 1: Get Diagnosed Conditions
    try:
        logger.info("Processing diagnosis...")
        if diagnosis_str and diagnosis_str != 'General condition':
            predicted_diagnosis = diagnosis_str
        else:
            # Try to get diagnosis from the diagnose function
            diagnosis_result = diagnose(symptoms, diagnosis, test_results)
            predicted_diagnosis = diagnosis_result.get("predicted_conditions", diagnosis_result.get("predicted_diagnosis", diagnosis_str))
        
        logger.info(f"Using diagnosis: {predicted_diagnosis}")
    except Exception as e:
        logger.error(f"Error processing diagnosis: {e}")
        predicted_diagnosis = diagnosis_str or "General medical condition"
    
    # Step 2: Construct patient data summary
    patient_data = f"""
    Symptoms: {symptoms}
    Medical History: {diagnosis}  # Use diagnosis to represent the patient's history
    Test Results: {test_results}
    Predicted Diagnosis: {predicted_diagnosis}
    """
    logger.info(f"Patient Data: {patient_data}")
    
    # Step 3: Generate Initial Follow-Up Plan with BioGPT
    input_text = f"""
    Create a detailed follow-up plan for a patient with the following medical details:
    {patient_data}
    
    Ensure the plan covers:
    - Week-wise monitoring of symptoms and treatments
    - Lifestyle recommendations
    - Necessary medical tests
    - Early warning signs to track
    """
    
    # Ensure the input text is within model’s max length
    # Generate follow-up plan using available resources
    if MODEL and TOKENIZER:
        inputs = TOKENIZER(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    else:
        # Use Cohere directly if BioGPT is not available
        if co:
            try:
                cohere_response = co.chat(
                    model=COHERE_MODEL,
                    message=input_text,
                    temperature=COHERE_PARAMS["temperature"]
                )
                initial_followup = cohere_response.text.strip()
            except Exception as e:
                logger.error(f"Cohere error: {e}")
                initial_followup = f"Comprehensive follow-up plan for {predicted_diagnosis}: Weekly symptom monitoring, lifestyle modifications, and regular medical check-ups recommended."
        else:
            initial_followup = f"Standard follow-up plan for {predicted_diagnosis}: Monitor symptoms, follow medication regimen, and schedule regular appointments."
        
        return {
            "predicted_diagnosis": predicted_diagnosis,
            "initial_followup": initial_followup,
            "refined_followup": initial_followup
        }
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                output_tokens = MODEL.generate(
                    **inputs,
                    max_length=700,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    num_return_sequences=1
                )
            initial_followup = TOKENIZER.decode(output_tokens[0], skip_special_tokens=True)
            logger.info("Initial Follow-Up Plan Generated.")
        except Exception as e:
            logger.error(f"Error generating follow-up with BioGPT: {e}")
            initial_followup = f"General follow-up plan for {predicted_diagnosis}: Continue current treatment, monitor progress, and maintain regular medical supervision."
    
    # Step 4: Refine Plan with Cohere (If Available)
    if co:
        try:
            logger.info("Refining follow-up plan using Cohere...")
            cohere_prompt = f"""
            You are a medical assistant AI. Based on the patient's information below, refine the follow-up plan into a clear, week-wise structure:
            
            Patient History:
            {patient_data}
            
            Initial Follow-Up Plan:
            {initial_followup}
            
            Provide a structured follow-up plan with:
            1. Week-by-week breakdown
            2. Medical tests required
            3. Adjustments to medications or treatments
            4. Lifestyle and behavioral advice
            """
            
            cohere_response = co.chat(
                model=COHERE_MODEL,
                message=cohere_prompt,
                temperature=COHERE_PARAMS["temperature"]
            )
            refined_followup_plan = cohere_response.text.strip()
            logger.info("Refined Follow-Up Plan Generated.")
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            refined_followup_plan = f"Error generating refined follow-up. Using initial plan: \n\n{initial_followup}"
    else:
        refined_followup_plan = initial_followup
    
    logger.info("Final Follow-Up Plan Generated.")
    
    # Returning the final result as a dictionary
    return {
        "predicted_diagnosis": predicted_diagnosis,  # Use predicted_diagnosis here
        "initial_followup": initial_followup,
        "refined_followup": refined_followup_plan
    }
