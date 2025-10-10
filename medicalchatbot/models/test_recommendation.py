import os
import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import cohere
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COHERE_API_KEY, COHERE_MODEL, COHERE_PARAMS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Cohere API client
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# Define device for model loading
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Skip large model loading to prevent memory issues
logger.info("Skipping FLAN-T5 model loading to preserve memory for BioGPT")
FLAN_T5_MODEL = None
FLAN_T5_TOKENIZER = None

# Format previous test results into a readable string
def format_test_results(test_results):
    if isinstance(test_results, dict) and test_results:
        return '\n'.join([f"{key}: {value}" for key, value in test_results.items()])
    return "No test results"

# Parse raw recommendations into structured format
def parse_recommendations(raw_recommendations):
    recommendations_list = []
    if raw_recommendations:
        lines = raw_recommendations.split("\n")
        for line in lines:
            if line.strip():
                parts = line.split(" - ", 1)
                if len(parts) == 2:
                    test_name, test_description = parts
                    recommendations_list.append({
                        "test_name": test_name.strip(),
                        "test_description": test_description.strip()
                    })
    return recommendations_list

# Generate a detailed report for recommended tests
def generate_report_for_diagnosis(diagnosis, confidence, recommendations):
    report = f"Recommended Tests:\nBelow is a refined and prioritized list of tests recommended for a patient with {diagnosis} based on the provided factors:\n"
    for i, recommendation in enumerate(recommendations, 1):
        test_name = recommendation.get("test_name", "Unknown Test")
        test_description = recommendation.get("test_description", "No description available.")
        report += f"{i}. {test_name} - {test_description}\n"
    return report

# Generate test recommendations based on diagnoses and previous test results

def generate_test_recommendations(diagnosis, test_results):
    try:
        if not diagnosis or (isinstance(diagnosis, str) and not diagnosis.strip()):
            diagnosis = "General medical condition"
            logger.warning("No specific diagnosis provided, using general condition")

        formatted_results = format_test_results(test_results)
        if not formatted_results:
            raise ValueError("Test Results is missing or invalid.")

        logger.info("Generating test recommendations...")
        prompt = f"""
You are a medical expert providing comprehensive test recommendations.

Diagnosis: {diagnosis}
Previous Test Results: {formatted_results}

Provide detailed test recommendations with:

1. ESSENTIAL TESTS:
   - Specific test names
   - Clinical rationale for each test
   - Expected findings
   - Urgency level (Urgent/Routine)

2. ADDITIONAL TESTS (if indicated):
   - Confirmatory tests
   - Monitoring tests
   - Screening tests

3. IMAGING STUDIES (if needed):
   - Type of imaging
   - Specific views or protocols
   - Clinical indication

4. SPECIALIST CONSULTATIONS:
   - Which specialists to consult
   - Reason for referral

Format as a structured medical recommendation with clear sections and explanations.
"""

        # Use Cohere for detailed recommendations
        if co:
            cohere_response = co.chat(
                model=COHERE_MODEL,
                message=prompt.strip(),
                                                                                                                                               temperature=0.3
            )
            recommendation = cohere_response.text.strip()
        else:
            recommendation = f"""
ESSENTIAL TESTS for {diagnosis}:

1. Complete Blood Count (CBC)
   - Rationale: Assess for infection, anemia, or blood disorders
   - Urgency: Routine

2. Basic Metabolic Panel (BMP)
   - Rationale: Evaluate kidney function and electrolyte balance
   - Urgency: Routine

3. Urinalysis
   - Rationale: Screen for kidney disease and urinary tract infections
   - Urgency: Routine

ADDITIONAL TESTS:
- Consider chest X-ray if respiratory symptoms present
- Lipid panel for cardiovascular risk assessment
"""

        # Add priority and safety information
        if co and recommendation:
            logger.info("Adding priority and safety information...")
            try:
                safety_response = co.chat(
                    model=COHERE_MODEL,
                    message=f"""
Review and enhance the following test recommendations with priority levels and safety considerations:

Diagnosis: {diagnosis}
Current Recommendations: {recommendation}

Add:
- Priority levels (High/Medium/Low)
- Safety precautions if any
- Cost-effectiveness notes
- Timeline for completion

Return the enhanced recommendations.
""",
                    temperature=0.2
                )
                enhanced = safety_response.text.strip()
                return enhanced if enhanced else recommendation
            except Exception as e:
                logger.error(f"Error enhancing recommendations: {e}")
                return recommendation

        return recommendation

    except Exception as e:
        logger.error(f"Error generating test recommendations: {e}")
        return f"Error generating test recommendations: {str(e)}"

