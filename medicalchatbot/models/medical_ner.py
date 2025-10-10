import re
import os
import sys
import json
import cohere
from typing import Dict, List, Set
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import COHERE_API_KEY, COHERE_MODEL

logger = get_logger(__name__)

class MedicalNER:
    def __init__(self):
        self.medical_entities = self._load_medical_entities()
        
    def _load_medical_entities(self) -> Dict[str, Set[str]]:
        """LLM will handle entity extraction dynamically"""
        return {}
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """LLM-powered medical entity extraction"""
        co = cohere.Client(COHERE_API_KEY)
        
        response = co.chat(
            model=COHERE_MODEL,
            message=f"Extract medical entities from: {text}",
            preamble="""You are a medical NLP AI. Extract medical entities and respond in JSON format:
{
  "symptoms": ["list of symptoms"],
  "conditions": ["list of medical conditions"], 
  "medications": ["list of drugs/medications"],
  "tests": ["list of medical tests"],
  "body_parts": ["list of body parts mentioned"]
}
Only include entities actually mentioned in the text.""",
            temperature=0.1
        )
        
        try:
            result = json.loads(response.text)
            return result
        except:
            return {"symptoms": [], "conditions": [], "medications": [], "tests": [], "body_parts": []}
    
    def extract_dosages(self, text: str) -> List[Dict]:
        """LLM-powered dosage extraction"""
        co = cohere.Client(COHERE_API_KEY)
        
        response = co.chat(
            model=COHERE_MODEL,
            message=f"Extract medication dosages from: {text}",
            preamble="""Extract medication dosages and respond in JSON format:
{
  "dosages": [
    {
      "medication": "drug name",
      "dose": "amount",
      "unit": "mg/g/ml/etc",
      "frequency": "daily/twice/etc"
    }
  ]
}""",
            temperature=0.1
        )
        
        try:
            result = json.loads(response.text)
            return result.get("dosages", [])
        except:
            return []
    
    def extract_vital_signs(self, text: str) -> Dict[str, str]:
        """Extract vital signs from text"""
        vital_patterns = {
            "blood_pressure": r'(?:bp|blood pressure)[\s:]*(\d{2,3})/(\d{2,3})',
            "heart_rate": r'(?:hr|heart rate|pulse)[\s:]*(\d{2,3})\s*(?:bpm)?',
            "temperature": r'(?:temp|temperature)[\s:]*(\d{2,3}(?:\.\d)?)\s*(?:°?f|°?c)?',
            "respiratory_rate": r'(?:rr|respiratory rate)[\s:]*(\d{1,2})',
            "oxygen_saturation": r'(?:o2 sat|spo2|oxygen)[\s:]*(\d{2,3})%?'
        }
        
        vitals = {}
        for vital, pattern in vital_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if vital == "blood_pressure":
                    vitals[vital] = f"{match.group(1)}/{match.group(2)}"
                else:
                    vitals[vital] = match.group(1)
        
        return vitals
    
    def get_comprehensive_analysis(self, text: str) -> Dict:
        """Get comprehensive medical entity analysis"""
        entities = self.extract_entities(text)
        dosages = self.extract_dosages(text)
        vitals = self.extract_vital_signs(text)
        
        # Count entities
        entity_counts = {category: len(items) for category, items in entities.items() if items}
        
        analysis = {
            "entities": entities,
            "dosages": dosages,
            "vital_signs": vitals,
            "entity_counts": entity_counts,
            "summary": self._generate_summary(entities, dosages, vitals)
        }
        
        return analysis
    
    def _generate_summary(self, entities: Dict, dosages: List, vitals: Dict) -> str:
        """Generate summary of extracted entities"""
        summary_parts = []
        
        for category, items in entities.items():
            if items:
                summary_parts.append(f"{len(items)} {category}")
        
        if dosages:
            summary_parts.append(f"{len(dosages)} dosages")
        
        if vitals:
            summary_parts.append(f"{len(vitals)} vital signs")
        
        if summary_parts:
            return f"Extracted: {', '.join(summary_parts)}"
        else:
            return "No medical entities detected"

# Initialize global NER
medical_ner = MedicalNER()