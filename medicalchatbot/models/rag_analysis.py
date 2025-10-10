import torch
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Advanced RAG configuration
RAG_CONFIG = {
    'max_context_length': 2048,
    'min_confidence_threshold': 0.6,
    'semantic_similarity_threshold': 0.75,
    'max_retrieved_docs': 5,
    'chunk_overlap': 50
}

# Load BioGPT model (optional)
def load_biogpt_model():
    """Load BioGPT model with fallback options"""
    models_to_try = [
        "microsoft/BioGPT-Large",
        "microsoft/BioGPT",
        "microsoft/DialoGPT-medium"  # Fallback
    ]
    
    for model_name in models_to_try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            logger.info(f"Model {model_name} loaded successfully on {device}.")
            return model, tokenizer, device
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            continue
    
    logger.warning("No suitable model could be loaded.")
    return None, None, None

# Load multiple embedding models for ensemble
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Medical-specific embedding model
    medical_embedding = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    logger.info("Embedding models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading embedding models: {e}")
    raise RuntimeError("Failed to load embedding models.") from e

# Initialize BioGPT model (optional)
biogpt_model, biogpt_tokenizer, biogpt_device = load_biogpt_model()

def extract_medical_keywords(text: str) -> List[str]:
    """Extract medical keywords for better context matching"""
    medical_patterns = [
        r'\b(?:symptom|diagnosis|treatment|medication|disease|condition)s?\b',
        r'\b(?:pain|fever|headache|nausea|fatigue|cough)\b',
        r'\b(?:mg|ml|tablet|capsule|dose|dosage)\b',
        r'\b(?:blood|urine|x-ray|mri|ct|scan)\b'
    ]
    
    keywords = []
    for pattern in medical_patterns:
        matches = re.findall(pattern, text.lower())
        keywords.extend(matches)
    
    return list(set(keywords))

def calculate_semantic_similarity(query: str, document: str) -> float:
    """Calculate semantic similarity between query and document"""
    try:
        query_embedding = sentence_transformer.encode([query])
        doc_embedding = sentence_transformer.encode([document])
        
        # Cosine similarity
        similarity = np.dot(query_embedding[0], doc_embedding[0]) / (
            np.linalg.norm(query_embedding[0]) * np.linalg.norm(doc_embedding[0])
        )
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def smart_chunk_text(text: str, max_length: int = 512) -> List[str]:
    """Intelligently chunk text preserving medical context"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def retrieve_relevant_data(query: str, faiss_index, num_results: int = 5, relevance_threshold: float = 0.7) -> Tuple[List[str], List[float]]:
    """Enhanced retrieval with confidence scoring and semantic filtering"""
    try:
        if not isinstance(faiss_index, FAISS):
            raise ValueError("Invalid FAISS index provided.")
        
        # Extract medical keywords from query
        query_keywords = extract_medical_keywords(query)
        logger.info(f"Extracted keywords: {query_keywords}")
        
        # Perform similarity search with scores
        search_results = faiss_index.similarity_search_with_score(query, k=num_results)
        
        filtered_results = []
        confidence_scores = []
        
        for doc, score in search_results:
            # Calculate semantic similarity
            semantic_sim = calculate_semantic_similarity(query, doc.page_content)
            
            # Boost score if medical keywords match
            keyword_boost = 0.0
            doc_keywords = extract_medical_keywords(doc.page_content)
            common_keywords = set(query_keywords) & set(doc_keywords)
            if common_keywords:
                keyword_boost = len(common_keywords) * 0.1
            
            # Combined confidence score
            combined_score = semantic_sim + keyword_boost
            
            if combined_score >= relevance_threshold:
                # Smart chunking for long documents
                chunks = smart_chunk_text(doc.page_content)
                filtered_results.extend(chunks[:2])  # Take top 2 chunks
                confidence_scores.extend([combined_score] * len(chunks[:2]))
        
        if not filtered_results:
            filtered_results = ["No highly relevant medical information found."]
            confidence_scores = [0.1]
        
        logger.info(f"Retrieved {len(filtered_results)} relevant chunks with avg confidence: {np.mean(confidence_scores):.2f}")
        return filtered_results, confidence_scores
        
    except Exception as e:
        logger.error(f"Error during enhanced FAISS retrieval: {e}")
        return ["Error retrieving information."], [0.0]

def generate_confidence_score(retrieved_docs: List[str], confidence_scores: List[float], query: str) -> float:
    """Generate overall confidence score for the response"""
    if not confidence_scores:
        return 0.1
    
    # Base confidence from retrieval scores
    base_confidence = np.mean(confidence_scores)
    
    # Boost confidence if multiple relevant documents found
    doc_count_boost = min(len(retrieved_docs) * 0.1, 0.3)
    
    # Reduce confidence for complex queries
    complex_terms = ['rare', 'unusual', 'complex', 'multiple conditions']
    complexity_penalty = 0.2 if any(term in query.lower() for term in complex_terms) else 0.0
    
    final_confidence = min(base_confidence + doc_count_boost - complexity_penalty, 1.0)
    return max(final_confidence, 0.1)

def generate_rag_response(query: str, faiss_index) -> Dict[str, any]:
    """Enhanced RAG response generation with confidence scoring and metadata"""
    try:
        # Validate query
        if not query or not isinstance(query, str) or not query.strip():
            return {
                "response": "Invalid query provided.",
                "confidence": 0.0,
                "sources_used": 0,
                "medical_keywords": []
            }
        
        # Extract medical context
        medical_keywords = extract_medical_keywords(query)
        
        # Retrieve relevant documents with confidence scores
        retrieved_docs, confidence_scores = retrieve_relevant_data(
            query, faiss_index, 
            num_results=RAG_CONFIG['max_retrieved_docs'],
            relevance_threshold=RAG_CONFIG['semantic_similarity_threshold']
        )
        
        # Generate overall confidence
        overall_confidence = generate_confidence_score(retrieved_docs, confidence_scores, query)
        
        # Prepare context for generation
        context = " ".join(retrieved_docs[:3])  # Use top 3 most relevant
        
        # If BioGPT is available, use it for generation
        if biogpt_model and biogpt_tokenizer:
            input_text = f"""Medical Context: {context}
Medical Keywords: {', '.join(medical_keywords)}
Question: {query}
Provide a concise, evidence-based medical response:"""
            
            inputs = biogpt_tokenizer(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=RAG_CONFIG['max_context_length'], 
                padding=True
            ).to(biogpt_device)
            
            with torch.no_grad():
                output_tokens = biogpt_model.generate(
                    **inputs,
                    max_length=600,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.9
                )
            
            generated_text = biogpt_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            # Extract only the answer part
            if "Provide a concise" in generated_text:
                response_text = generated_text.split("Provide a concise")[-1].strip()
            else:
                response_text = generated_text
                
        else:
            # Enhanced fallback response
            if overall_confidence > RAG_CONFIG['min_confidence_threshold']:
                response_text = f"Based on medical literature: {' '.join(retrieved_docs[:2])}"
            else:
                response_text = f"Limited information available. Context: {' '.join(retrieved_docs[:1])}"
        
        # Add confidence indicator
        confidence_emoji = "🟢" if overall_confidence > 0.8 else "🟡" if overall_confidence > 0.6 else "🔴"
        
        return {
            "response": f"{confidence_emoji} {response_text}",
            "confidence": overall_confidence,
            "sources_used": len(retrieved_docs),
            "medical_keywords": medical_keywords,
            "retrieval_scores": confidence_scores[:3],
            "context_length": len(context)
        }
    
    except Exception as e:
        logger.error(f"Error generating enhanced RAG response: {e}")
        return {
            "response": "Error generating medical response. Please try again.",
            "confidence": 0.0,
            "sources_used": 0,
            "medical_keywords": [],
            "error": str(e)
        }

