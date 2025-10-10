import logging
from models.diagnosis import diagnose
from models.treatment_recommendation import generate_treatment_plan
from models.followup import generate_followup_plan
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_conclusion(diagnosis, treatment_plan, follow_up_plan=None):
    """
    Generates a conclusion based on the predicted diagnosis and treatment plan.

    Args:
    - diagnosis (str): The predicted diagnosis of the patient.
    - treatment_plan (str): The generated treatment plan.
    - follow_up_plan (str, optional): Additional follow-up steps if provided.

    Returns:
    - str: The conclusion text that summarizes the treatment and follow-up plan.
    """
    try:
        conclusion = f"""
- Based on the predicted diagnosis of {diagnosis}, this treatment plan addresses the immediate concerns and focuses on managing the patient's symptoms.
- The prescribed medications, lifestyle changes, and week-wise treatment schedule aim to improve patient outcomes and ensure regular monitoring.
- Follow-up tests are recommended to ensure that the condition is progressing as expected, and further evaluation may be required if the patient's condition worsens.
"""
        return conclusion.strip()

    except Exception as e:
        logger.error(f"Error generating conclusion: {e}")
        return f"Error generating conclusion: {str(e)}"