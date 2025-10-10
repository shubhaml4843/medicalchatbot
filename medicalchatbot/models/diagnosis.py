import os
import logging
import cohere
from models.extract import extract_report_data
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import COHERE_API_KEY, COHERE_MODEL, COHERE_PARAMS

logger = get_logger(__name__)
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

def diagnose(report_text):
    extracted = extract_report_data(report_text)

    symptoms = ", ".join(extracted.get("symptoms", [])) or "No symptoms"
    vitals_dict = extracted.get("vital_signs", {})
    test_dict = extracted.get("test_results", {})
    vitals = ", ".join([f"{k}: {v}" for k, v in vitals_dict.items()]) or "No vitals"
    tests = ", ".join([f"{k}: {v}" for k, v in test_dict.items()]) or "No test results"
    prelim_diag = extracted.get("preliminary_diagnosis", "").strip()

    logger.info(f"Symptoms: {symptoms}")
    logger.info(f"Vitals: {vitals}")
    logger.info(f"Tests: {tests}")
    logger.info(f"Preliminary Diagnosis: {prelim_diag or 'Unknown'}")

    if not co:
        return {
            "predicted_conditions": [{"diagnosis": prelim_diag or "Unknown", "confidence": "Unknown"}],
            "explanation": "Cohere API key not set."
        }

    try:
        prompt = f"""
        You are an expert medical diagnostician. Analyze the following patient data:
        
        PATIENT SYMPTOMS: {symptoms}
        VITAL SIGNS: {vitals}
        LABORATORY/TEST RESULTS: {tests}
        PRELIMINARY ASSESSMENT: {prelim_diag}
        
        Provide differential diagnosis with confidence scores. Consider:
        - Symptom patterns and combinations
        - Vital sign abnormalities
        - Laboratory value interpretations
        - Age and demographic factors
        
        Format your response as:
        1. Primary Diagnosis - 85%
        2. Secondary Diagnosis - 10%
        3. Alternative Diagnosis - 5%
        
        Base confidence on clinical evidence strength.
        """

        response = co.chat(
            model=COHERE_MODEL,
            message=prompt.strip(),
            temperature=COHERE_PARAMS["temperature"]
        )

        raw_text = response.text.strip()
        logger.info(f"Raw Diagnosis Output:\n{raw_text}")

        # Parse the output into structured format
        diagnoses = []
        for line in raw_text.split("\n"):
            if "-" in line:
                try:
                    name, confidence = line.split("-", 1)
                    diagnosis = name.strip("1234567890. ").strip()
                    confidence = confidence.strip()
                    if diagnosis and confidence:
                        diagnoses.append({"diagnosis": diagnosis, "confidence": confidence})
                except Exception as e:
                    logger.warning(f"Failed to parse line: '{line}'. Error: {e}")

        if not diagnoses:
            diagnoses = [{"diagnosis": prelim_diag or "Unknown", "confidence": "Unknown"}]

        # Optional explanation for the top diagnosis
        top_diag = diagnoses[0]["diagnosis"]
        explanation = "No explanation generated."
        try:
            explanation_prompt = f"""
            Patient Symptoms: {symptoms}
            Vital Signs: {vitals}
            Test Results: {tests}
            Top Diagnosis: {top_diag}

            Explain why this diagnosis is probable.
            """
            explanation_response = co.chat(
                model=COHERE_MODEL,
                                                 message=explanation_prompt.strip(),
                temperature=0.6
            )
            explanation = explanation_response.text.strip()
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            explanation = "Explanation generation failed."

        return {
            "predicted_conditions": diagnoses,
            "explanation": explanation
        }

    except Exception as e:
        logger.error(f"Cohere failed to generate diagnosis: {e}")
        return {
            "predicted_conditions": [{"diagnosis": prelim_diag or "Unknown", "confidence": "Unknown"}],
            "explanation": "Cohere model failed. Returning fallback diagnosis."
        }
