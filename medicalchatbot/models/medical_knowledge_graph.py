import os
import sys
import json
import cohere
from typing import Dict, List, Set, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import COHERE_API_KEY, COHERE_MODEL

logger = get_logger(__name__)

class MedicalKnowledgeGraph:
    def __init__(self):
        self.knowledge_graph = self._build_knowledge_graph()
        
    def _build_knowledge_graph(self) -> Dict:
        """LLM will handle medical reasoning dynamically"""
        return {}
    
    def get_related_conditions(self, symptoms: List[str]) -> List[Dict]:
        """Get possible conditions based on symptoms"""
        condition_scores = {}
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower in self.knowledge_graph["symptoms_to_conditions"]:
                conditions = self.knowledge_graph["symptoms_to_conditions"][symptom_lower]
                for condition in conditions:
                    if condition not in condition_scores:
                        condition_scores[condition] = {"score": 0, "supporting_symptoms": []}
                    condition_scores[condition]["score"] += 1
                    condition_scores[condition]["supporting_symptoms"].append(symptom)
        
        # Sort by score and return top conditions
        sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        return [
            {
                "condition": condition,
                "confidence": min(data["score"] / len(symptoms), 1.0),
                "supporting_symptoms": data["supporting_symptoms"]
            }
            for condition, data in sorted_conditions[:5]
        ]
    
    def get_recommended_tests(self, conditions: List[str]) -> List[str]:
        """Get recommended tests for given conditions"""
        tests = set()
        
        for condition in conditions:
            condition_lower = condition.lower()
            if condition_lower in self.knowledge_graph["conditions_to_tests"]:
                tests.update(self.knowledge_graph["conditions_to_tests"][condition_lower])
        
        return list(tests)
    
    def get_treatment_options(self, conditions: List[str]) -> Dict[str, List[str]]:
        """Get treatment options for given conditions"""
        treatments = {}
        
        for condition in conditions:
            condition_lower = condition.lower()
            if condition_lower in self.knowledge_graph["conditions_to_treatments"]:
                treatments[condition] = self.knowledge_graph["conditions_to_treatments"][condition_lower]
        
        return treatments
    
    def get_drug_indications(self, drug: str) -> List[str]:
        """Get conditions that a drug can treat"""
        drug_lower = drug.lower()
        return self.knowledge_graph["drugs_to_conditions"].get(drug_lower, [])
    
    def get_risk_factors(self, condition: str) -> List[str]:
        """Get risk factors for a condition"""
        condition_lower = condition.lower()
        return self.knowledge_graph["risk_factors"].get(condition_lower, [])
    
    def generate_clinical_reasoning(self, symptoms: List[str], existing_conditions: List[str] = None) -> str:
        """LLM-powered clinical reasoning"""
        if not symptoms:
            return "No symptoms provided for analysis"
        
        co = cohere.Client(COHERE_API_KEY)
        
        symptoms_text = ", ".join(symptoms)
        conditions_text = ", ".join(existing_conditions) if existing_conditions else "None"
        
        response = co.chat(
            model=COHERE_MODEL,
            message=f"Provide clinical reasoning for symptoms: {symptoms_text}. Existing conditions: {conditions_text}",
            preamble="""You are a clinical reasoning AI. Provide structured medical analysis:

🧠 Clinical Reasoning:

📋 Differential Diagnoses:
1. [Most likely condition] (High probability)
2. [Second likely condition] (Medium probability) 
3. [Third likely condition] (Low probability)

🔬 Recommended Tests:
• [Test 1]
• [Test 2]
• [Test 3]

💊 Treatment Considerations:
• [Treatment approach 1]
• [Treatment approach 2]

Base recommendations on evidence-based medicine.""",
            temperature=0.2
        )
        
        return response.text.strip()
    
    def get_comprehensive_analysis(self, symptoms: List[str], conditions: List[str], medications: List[str]) -> Dict:
        """Get comprehensive medical analysis"""
        analysis = {
            "differential_diagnosis": self.get_related_conditions(symptoms),
            "recommended_tests": self.get_recommended_tests(conditions) if conditions else [],
            "treatment_options": self.get_treatment_options(conditions) if conditions else {},
            "drug_indications": {drug: self.get_drug_indications(drug) for drug in medications},
            "clinical_reasoning": self.generate_clinical_reasoning(symptoms, conditions)
        }
        
        return analysis

# Initialize global knowledge graph
medical_kg = MedicalKnowledgeGraph()