
import os
import logging
import re
import time
from fpdf import FPDF

from models.diagnosis import diagnose
from models.test_recommendation import generate_test_recommendations
from models.treatment_recommendation import generate_treatment_plan
from models.followup import generate_followup_plan
from models.generate_conclusion import generate_conclusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to generate the PDF report
def generate_pdf_report(patient_info):
    try:
        logger.info("Using structured patient information...")

        # Ensure name is properly formatted
        patient_info['name'] = str(patient_info.get('name', 'Unknown')).split('\n')[0].strip()

        required_keys = ['name', 'age', 'gender', 'symptoms', 'test_results', 'vital_signs']
        missing_keys = [key for key in required_keys if key not in patient_info]
        if missing_keys:
            raise ValueError(f"Missing required patient information: {', '.join(missing_keys)}")

        report_text = patient_info.get("full_report", "")

        logger.info("Predicting diagnosis using full report text...")
        diagnosis_result = diagnose(report_text)

        predicted_diagnosis = diagnosis_result.get("predicted_conditions", "Unknown Diagnosis")
        logger.info(f"Predicted Diagnosis: {predicted_diagnosis}")

        logger.info("Generating Test Recommendations...")
        test_results_str = format_test_results(patient_info.get('test_results', {}))
        test_recommendations_raw = generate_test_recommendations(predicted_diagnosis, test_results_str) or "No recommendations available"
        test_recommendations = remove_emojis(str(test_recommendations_raw))
        
        logger.info("Generating Treatment Plan...")
        treatment_plan_raw = generate_treatment_plan(predicted_diagnosis, patient_info) or "No treatment plan available."
        # Remove emojis and special characters for PDF compatibility
        treatment_plan = remove_emojis(str(treatment_plan_raw))
        
        logger.info("Generating Follow-Up Plan...")
        followup_data = generate_followup_plan(patient_info.get('symptoms', []), patient_info, patient_info.get('test_results', {})) or {}
        followup_plan_raw = followup_data.get("refined_followup", "No follow-up plan generated.")
        followup_plan = remove_emojis(str(followup_plan_raw))
        
        logger.info("Generating Conclusion...")
        conclusion_raw = generate_conclusion(predicted_diagnosis, treatment_plan, follow_up_plan=followup_plan)
        conclusion = remove_emojis(str(conclusion_raw))

        symptoms = ', '.join(patient_info['symptoms']) if isinstance(patient_info['symptoms'], list) else str(patient_info.get('symptoms', 'No symptoms recorded'))
        medications = ', '.join(patient_info.get('medications', [])) if isinstance(patient_info.get('medications'), list) else str(patient_info.get('medications', 'No medications recorded'))
        
        # Ensure all required fields have default values
        patient_info.setdefault('medical_history', 'No medical history available')
        patient_info.setdefault('medications', 'No medications recorded')

        logger.info("Formatting medical report...")

        pdf = FPDF()
        pdf.add_page()

        # Set font for the report title (bold)
        pdf.set_font("Arial", 'B', size=16)  # 'B' stands for bold
        pdf.cell(200, 10, txt="Medical Report", ln=True, align="C")
        pdf.ln(10)

        # Set font for the other titles (bold)
        pdf.set_font("Arial", 'B', size=12)

        # Patient Name Title
        pdf.multi_cell(0, 10, "Patient Name:")
        pdf.set_font("Arial", size=12)  # Resetting font to normal for content
        pdf.multi_cell(0, 10, f"{patient_info['name']}")
        pdf.ln(2)

        # Age Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Age:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{patient_info['age']}")
        pdf.ln(2)

        # Gender Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Gender:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{patient_info['gender']}")
        pdf.ln(2)

        # Medical History Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Medical History:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{patient_info.get('medical_history', 'No medical history available')}")
        pdf.ln(2)

        # Symptoms Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Symptoms:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{symptoms}")
        pdf.ln(2)

        # Test Results Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Test Results:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{format_test_results(patient_info['test_results'])}")
        pdf.ln(2)

        # Vital Signs Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Vital Signs:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{format_vital_signs(patient_info['vital_signs'])}")
        pdf.ln(2)

        # Medications Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Medications:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{medications}")
        pdf.ln(2)

        # Diagnosis Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Diagnosis:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{predicted_diagnosis}")
        pdf.ln(2)

        # Recommended Tests Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Recommended Tests:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{test_recommendations}")
        pdf.ln(2)

        # Treatment Plan Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Treatment Plan:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{treatment_plan}")
        pdf.ln(2)

        # Follow-Up Plan Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Follow-Up Plan:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{followup_plan}")
        pdf.ln(2)

        # Conclusion Title
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 10, "Conclusion:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"{conclusion}")
        pdf.ln(10)

        # Secure file path handling
        reports_dir = os.path.abspath("reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Sanitize filename
        safe_name = re.sub(r'[^\w\s-]', '', patient_info['name']).strip().replace(' ', '_')
        if not safe_name:
            safe_name = "patient"
        
        filename = f"report_{safe_name}_{int(time.time())}.pdf"
        file_path = os.path.join(reports_dir, filename)
        
        # Validate path is within reports directory
        if not os.path.abspath(file_path).startswith(reports_dir):
            raise ValueError("Invalid file path detected")
        
        pdf.output(file_path)

        logger.info(f"PDF successfully generated at: {file_path}")
        return os.path.abspath(file_path)

    except Exception as e:
        logger.error(f"Error in generating the medical report: {e}")
        raise e

# Format Test Results
def format_test_results(test_results):
    if isinstance(test_results, dict):
        return '\n'.join([f"- {key}: {value}" for key, value in test_results.items()])
    return "No test results available."

def remove_emojis(text):
    """Remove emojis and special Unicode characters for PDF compatibility"""
    import re
    if not text:
        return text
    
    # Replace smart quotes and special characters with ASCII equivalents
    replacements = {
        '\u2019': "'",  # Right single quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Horizontal ellipsis
        '\u00a0': ' ',  # Non-breaking space
    }
    
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Remove emojis and other special Unicode characters
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001F900-\U0001F9FF"  # supplemental symbols
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove any remaining non-ASCII characters that might cause issues
    text = ''.join(char if ord(char) < 128 else '?' for char in text)
    
    return text

# Format Vital Signs
def format_vital_signs(vital_signs):
    if isinstance(vital_signs, dict):
        return '\n'.join([f"- {key}: {value}" for key, value in vital_signs.items()])
    return "No vital signs available."
