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

def generate_diagnosis_specific_plan(diagnosis, symptoms):
    if co:
        try:
            prompt = f"Create a specific follow-up plan for: {diagnosis}\nPatient symptoms: {symptoms}\n\nFormat as:\n1st Week:\n- [specific actions for this diagnosis]\n\n2nd Week:\n- [specific follow-up for this condition]\n\n3rd-4th Week:\n- [recovery/monitoring for this diagnosis]\n\nMonthly:\n- [long-term care for this condition]\n\nMake it specific to {diagnosis}, not generic."
            
            response = co.chat(
                model=COHERE_MODEL,
                message=prompt,
                temperature=0.7
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"AI plan generation failed: {e}")
    
    return f"""1st Week:\n- Monitor {diagnosis} symptoms daily\n- Follow prescribed treatment\n- Track progress closely\n\n2nd Week:\n- Medical follow-up for {diagnosis}\n- Review treatment effectiveness\n- Adjust therapy if needed\n\n3rd-4th Week:\n- Continue {diagnosis} monitoring\n- Gradual activity resumption\n- Complete recommended tests\n\nMonthly:\n- Comprehensive {diagnosis} evaluation\n- Long-term management planning\n- Preventive care measures"""

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
    # Initialize initial_followup variable
    initial_followup = f"Basic follow-up plan for {predicted_diagnosis}: Regular monitoring and medical check-ups recommended."
    
    # Generate follow-up plan using available resources
    if MODEL and TOKENIZER:
        inputs = TOKENIZER(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
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
            initial_followup = generate_diagnosis_specific_plan(predicted_diagnosis, symptoms)
        
        return {
            "predicted_diagnosis": predicted_diagnosis,
            "initial_followup": initial_followup,
            "refined_followup": initial_followup
        }
        
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
            initial_followup = f"""
**Week 1-2:**
- Monitor {predicted_diagnosis} symptoms daily
- Continue current treatment regimen
- Track vital signs if applicable
- Report any worsening to healthcare provider

**Week 3-4:**
- Schedule follow-up appointment
- Review treatment effectiveness
- Adjust medications if needed
- Begin gradual activity increase

**Month 2:**
- Comprehensive health assessment
- Laboratory tests if required
- Long-term management planning
- Lifestyle modification counseling
"""
    
    # Step 4: Refine Plan with Cohere (If Available)
    if co:
        try:
            logger.info("Refining follow-up plan using Cohere...")
            cohere_prompt = f"""
            You are a medical assistant AI. Create a detailed follow-up plan with specific timeframes:
            
            Patient Condition: {predicted_diagnosis}
            Symptoms: {symptoms}
            Test Results: {test_results}
            
            Create a structured follow-up plan with:
            
            **Week 1:**
            - Daily monitoring requirements
            - Immediate actions needed
            - Medication schedule
            
            **Week 2:**
            - Progress assessment
            - Symptom tracking
            - Follow-up tests if needed
            
            **Week 3-4:**
            - Long-term monitoring
            - Lifestyle modifications
            - Next appointment scheduling
            
            **Monthly Follow-up:**
            - Regular check-ups
            - Preventive measures
            - Warning signs to watch
            
            Provide specific dates, times, and actionable steps.
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
