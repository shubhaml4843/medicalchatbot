import json
import os
import sys
import cohere
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import COHERE_API_KEY, COHERE_MODEL

logger = get_logger(__name__) 

class DrugInteractionChecker:
    def __init__(self):
        self.interactions = self._load_drug_interactions()
        
    def _load_drug_interactions(self) -> Dict:
        """LLM-powered drug interaction detection"""
        return {}  # Will use LLM for dynamic detection
    
    def check_interactions(self, medications: List[str]) -> List[Dict]:
        """LLM-powered drug interaction checking"""
        if not medications or len(medications) < 2:
            return []
        co = cohere.Client(COHERE_API_KEY)
        
        med_list = ", ".join(medications)
        
        response = co.chat(
            model=COHERE_MODEL,
            message=f"Analyze drug interactions for: {med_list}",
            preamble="""You are a clinical pharmacist AI. Analyze drug interactions and respond in JSON format:
{
  "interactions": [
    {
      "drug1": "drug name",
      "drug2": "drug name", 
      "severity": "HIGH/MEDIUM/LOW",
      "risk": "description of risk",
      "recommendation": "clinical recommendation"
    }
  ]
}
Only include actual interactions. If no interactions, return empty array.""",
            temperature=0.1
        )
        
        try:
            result = json.loads(response.text)
            return result.get("interactions", [])
        except:
            return []
    
    def assess_symptom_severity(self, symptoms: List[str], patient_data: Dict) -> Dict:
        """LLM-powered symptom severity assessment"""
        co = cohere.Client(COHERE_API_KEY)
        
        response = co.chat(
            model=COHERE_MODEL,
            message=f"Assess severity: {', '.join(symptoms)}. Patient: {patient_data}",
            preamble="""Rate symptom severity in JSON format:
{
  "overall_severity": "CRITICAL/HIGH/MEDIUM/LOW",
  "urgency": "IMMEDIATE/URGENT/ROUTINE",
  "triage_level": "ER/URGENT_CARE/CLINIC",
  "explanation": "clinical reasoning"
}""",
            temperature=0.1
        )
        
        try:
            return json.loads(response.text)
        except:
            return {"overall_severity": "UNKNOWN", "urgency": "ROUTINE"}
    
    def get_personalized_recommendations(self, age: int, weight: float, conditions: List[str]) -> Dict:
        """Age/weight-based personalized recommendations"""
        co = cohere.Client(COHERE_API_KEY)
        
        response = co.chat(
            model=COHERE_MODEL,
            message=f"Personalized recommendations for {age}yo, {weight}kg patient with {conditions}",
            preamble="""Provide personalized medical recommendations in JSON:
{
  "dosage_adjustments": ["specific adjustments"],
  "contraindications": ["age/weight specific warnings"],
  "monitoring": ["what to monitor"],
  "lifestyle": ["personalized advice"]
}""",
            temperature=0.1
        )
        
        try:
            return json.loads(response.text)
        except:
            return {}
    
    def clinical_decision_support(self, symptoms: List[str], vitals: Dict, history: List[str]) -> Dict:
        """Comprehensive clinical decision support"""
        co = cohere.Client(COHERE_API_KEY)
        
        response = co.chat(
            model=COHERE_MODEL,
            message=f"Clinical decision support: Symptoms: {symptoms}, Vitals: {vitals}, History: {history}",
            preamble="""Provide clinical decision support in JSON:
{
  "primary_diagnosis": "most likely diagnosis",
  "differential": ["other possibilities"],
  "immediate_actions": ["urgent steps"],
  "diagnostic_tests": ["recommended tests"],
  "treatment_plan": ["treatment steps"],
  "red_flags": ["warning signs"],
  "follow_up": "follow-up plan"
}""",
            temperature=0.1
        )
        
        try:
            return json.loads(response.text)
        except:
            return {}
    
    def calculate_risk_scores(self, patient_data: Dict) -> Dict:
        """Calculate medical risk scores"""
        co = cohere.Client(COHERE_API_KEY)
        
        response = co.chat(
            model=COHERE_MODEL,
            message=f"Calculate risk scores for patient: {patient_data}",
            preamble="""Calculate relevant medical risk scores in JSON:
{
  "cardiovascular_risk": {"score": "percentage", "category": "low/medium/high"},
  "bleeding_risk": {"score": "percentage", "category": "low/medium/high"},
  "fall_risk": {"score": "percentage", "category": "low/medium/high"},
  "recommendations": ["risk-based recommendations"]
}""",
            temperature=0.1
        )
        
        try:
            return json.loads(response.text)
        except:
            return {}
    
    def search_latest_research(self, condition: str, treatment: str) -> str:
        """Search for latest medical research"""
        co = cohere.Client(COHERE_API_KEY)
        
        response = co.chat(
            model=COHERE_MODEL,
            message=f"Latest research on {condition} treatment with {treatment}",
            preamble="""Provide latest evidence-based research summary:
- Recent clinical trials
- Updated guidelines
- New treatment protocols
- Safety updates
Format as brief clinical summary.""",
            temperature=0.2
        )
        
        return response.text.strip()
    
    def get_safety_report(self, medications: List[str]) -> str:
        """Generate comprehensive safety report"""
        interactions = self.check_interactions(medications)
        
        if not interactions:
            return "✅ No known drug interactions detected"
        
        report = f"🔍 Drug Interaction Analysis ({len(interactions)} interactions found):\n\n"
        
        # Sort by severity
        high_risk = [i for i in interactions if i["severity"] == "HIGH"]
        medium_risk = [i for i in interactions if i["severity"] == "MEDIUM"]
        low_risk = [i for i in interactions if i["severity"] == "LOW"]
        
        for risk_level, interactions_list, emoji in [
            ("HIGH RISK", high_risk, "🚨"),
            ("MEDIUM RISK", medium_risk, "⚠️"),
            ("LOW RISK", low_risk, "ℹ️")
        ]:
            if interactions_list:
                report += f"{emoji} {risk_level} INTERACTIONS:\n"
                for interaction in interactions_list:
                    report += f"• {interaction['drug1']} + {interaction['drug2']}\n"
                    report += f"  Risk: {interaction['risk']}\n"
                    report += f"  Action: {interaction['recommendation']}\n\n"
        
        return report

# Initialize global advanced medical AI
medical_ai = DrugInteractionChecker()