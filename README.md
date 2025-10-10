# Medical Chatbot AI System

An advanced AI-powered medical chatbot system that provides intelligent medical assistance, diagnosis support, treatment recommendations, and comprehensive medical report generation.

## 🚀 Features

### Core Capabilities
- **AI Medical Assistant**: Provides evidence-based medical recommendations
- **Patient Simulation**: Simulates patient responses for training scenarios
- **Medical Report Processing**: Extracts and analyzes clinical data from uploaded reports
- **Intelligent Diagnosis**: AI-powered differential diagnosis with confidence scoring
- **Treatment Planning**: Personalized treatment recommendations with drug interaction checks
- **Test Recommendations**: Suggests appropriate medical tests based on symptoms
- **Follow-up Planning**: Generates structured follow-up care plans
- **PDF Report Generation**: Creates comprehensive medical reports

### Advanced AI Features
- **RAG (Retrieval Augmented Generation)**: Enhanced responses using medical knowledge base via FAISS vector search
- **Medical NER (Named Entity Recognition)**: Automatically extracts symptoms, conditions, medications from clinical text
- **Medical Knowledge Graph**: Provides clinical reasoning and connects medical concepts
- **Drug Interaction Checking**: Real-time safety validation for medication combinations
- **Severity Assessment**: AI-powered triage and symptom severity analysis
- **Clinical Decision Support**: Evidence-based diagnostic and treatment recommendations
- **Personalized Dosing**: Age and weight-based medication dosage calculations
- **Confidence Scoring**: Provides reliability indicators for AI recommendations
- **Multi-Modal Analysis**: Combines text analysis with clinical reasoning

## 🛠️ Technology Stack

### AI Models
- **Cohere Command-A-03-2025**: Primary language model for medical reasoning and recommendations
- **BioGPT-Large**: Specialized biomedical text generation for follow-up plans
- **FLAN-T5-Large**: Treatment plan generation and medical advice
- **SentenceTransformers (all-MiniLM-L6-v2)**: Medical text embeddings for FAISS vector search
- **HuggingFace Transformers**: NLP model integration and medical entity recognition
- **Medical NER Models**: Custom trained models for clinical entity extraction

### Backend
- **Flask**: Web application framework
- **FAISS**: Vector similarity search for medical knowledge retrieval
- **LangChain**: AI application framework
- **PyTorch**: Deep learning framework

### Data Processing
- **Medical NER**: Named entity recognition for symptoms, conditions, medications, dosages
- **Document Processing**: PDF, DOCX, TXT file support with clinical data extraction
- **Text Chunking**: Intelligent medical text segmentation for FAISS indexing
- **Clinical Data Extraction**: Automated extraction of vital signs, lab results, medical history
- **Vector Embeddings**: Semantic search capabilities for medical documents
- **Knowledge Graph Processing**: Medical concept relationship mapping

## 📋 Prerequisites

- Python 3.11+
- CUDA-compatible GPU (optional, for faster processing)
- Cohere API Key

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd medicalchatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   COHERE_API_KEY=your_cohere_api_key_here
   FLASK_DEBUG=False
   SECRET_KEY=your_secret_key_here
   ```

## 🚀 Usage

### Starting the Application

**Standard Mode:**
```bash
python app.py
```

**Memory Optimized Mode:**
```bash
python run_with_memory.py
```

The application will be available at `http://127.0.0.1:8040`

### Using the Medical Chatbot

1. **Upload Medical Report**
   - Navigate to the web interface
   - Upload a medical report (PDF, DOCX, or TXT)
   - System will process and extract clinical data
   - FAISS index will be created for semantic search

2. **Interact with the Chatbot**
   
   **AI Mode (Medical Consultant)** - Triggered by:
   - "What diagnosis do you recommend?"
   - "What treatment should I prescribe?"
   - "What tests should I order?"
   - "I recommend...", "I suggest...", "I will give..."
   
   **Patient Mode (Patient Simulation)** - Triggered by:
   - "How are you feeling?"
   - "What's your name?", "How old are you?"
   - "Are you taking any medication?"
   - General conversation and personal questions

3. **Advanced Features**
   - **Drug Interaction Checking**: Automatic safety validation when multiple medications mentioned
   - **Severity Assessment**: AI determines urgency level (URGENT/ROUTINE/CLINIC)
   - **Clinical Reasoning**: Knowledge graph provides medical connections
   - **Entity Extraction**: Automatically identifies medical terms from conversation

4. **Generate Medical Reports**
   - Click "Generate Report" to create comprehensive PDF reports
   - Includes diagnosis, treatment plans, follow-up recommendations
   - Enhanced with AI analysis and safety assessments

## 📁 Project Structure

```
medicalchatbot/
├── config.py                 # Centralized configuration
├── app.py                    # Main Flask application
├── run_with_memory.py        # Memory-optimized launcher
├── requirements.txt          # Python dependencies
├── models/                   # AI model modules
│   ├── chatbot.py           # Main chatbot logic with AI/Patient mode classification
│   ├── diagnosis.py         # AI diagnosis generation with confidence scoring
│   ├── treatment_recommendation.py  # Treatment planning with drug interaction checks
│   ├── test_recommendation.py       # Test suggestions based on clinical data
│   ├── followup.py          # Follow-up care planning with BioGPT
│   ├── extract.py           # Medical data extraction from reports
│   ├── data_processing.py   # Data preprocessing and cleaning
│   ├── rag_analysis.py      # RAG implementation with FAISS
│   ├── report_generation.py # PDF report creation
│   ├── medical_ner.py       # Medical named entity recognition
│   ├── medical_knowledge_graph.py  # Clinical reasoning and knowledge connections
│   └── drug_interaction_checker.py # Drug safety and interaction analysis
├── utils/                   # Utility functions
│   └── logger.py           # Logging configuration
├── templates/              # HTML templates
│   └── index.html         # Web interface
├── uploads/               # Uploaded files (temporary)
├── reports/              # Generated PDF reports
├── faiss_index/         # Vector database storage
└── logs/               # Application logs
```

## ⚙️ Configuration

The system uses centralized configuration in `config.py`:

### AI Models
- **Cohere Model**: `command-a-03-2025` (best performance)
- **Device**: Auto-detects CUDA/CPU
- **Model Parameters**: Optimized for medical use

### Medical Configuration
- **Confidence Threshold**: 0.7 for diagnosis reliability
- **Max Medications**: 5 per recommendation
- **Drug Interaction Checking**: Enabled with safety reports
- **Context Length**: 256K tokens for comprehensive analysis
- **Severity Levels**: URGENT, ROUTINE, CLINIC triage classification
- **Entity Extraction**: Symptoms, conditions, medications, dosages
- **Clinical Reasoning**: Knowledge graph depth of 3 levels

### File Upload
- **Allowed Types**: PDF, DOCX, DOC, TXT
- **Max Size**: 16MB
- **Security**: File validation and sanitization

## 🔒 Security Features

- **Input Validation**: Comprehensive file and data validation
- **Path Traversal Protection**: Secure file handling
- **Session Management**: Secure user sessions
- **Rate Limiting**: API abuse prevention
- **Error Handling**: Graceful error management

## 🧪 API Endpoints

### File Upload
```http
POST /upload_report
Content-Type: multipart/form-data
```

### Chat Interface
```http
POST /ask
Content-Type: application/json
{
  "question": "What diagnosis do you recommend?"
}
```

### Report Generation
```http
POST /generate_report
```

## 📊 Performance Optimization

### Memory Management
- **Model Loading**: Optimized for low memory usage with selective loading
- **Garbage Collection**: Automatic memory cleanup after processing
- **CUDA Optimization**: GPU memory management for transformer models
- **Batch Processing**: Efficient text processing for large documents
- **Advanced Features**: Conditional loading of NER, Knowledge Graph, Drug Checker

### Caching
- **FAISS Index**: Persistent vector storage for fast document retrieval
- **Session Data**: Efficient clinical data management
- **Model Caching**: Reduced loading times for frequently used models
- **Entity Cache**: Cached medical entity extractions
- **Knowledge Graph Cache**: Pre-computed clinical reasoning paths

## 🐛 Troubleshooting

### Common Issues

**FAISS Index Warning**
```
WARNING: No FAISS index found
```
- **Solution**: Normal on first run, upload a medical report to create index
- **Note**: System creates empty index as fallback, upload document for full functionality

**Memory Issues**
```
CUDA out of memory
```
- **Solution**: Use `run_with_memory.py` or set `DEVICE="cpu"` in config

**API Errors**
```
Cohere API error
```
- **Solution**: Check COHERE_API_KEY in .env file
- **Rate Limits**: Cohere API has usage limits, check your account status

**Advanced Features Not Loading**
```
Advanced features not available
```
- **Solution**: Check if medical_ner.py, medical_knowledge_graph.py, drug_interaction_checker.py exist
- **Fallback**: System works with basic features if advanced modules unavailable

### Logs
Check application logs in the `logs/` directory for detailed error information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Cohere**: Advanced language model API for medical reasoning
- **Hugging Face**: Pre-trained model ecosystem and transformers
- **LangChain**: AI application framework and document processing
- **FAISS**: Efficient similarity search for medical document retrieval
- **Flask**: Web application framework
- **SentenceTransformers**: Semantic embeddings for medical text
- **PyTorch**: Deep learning framework for model inference
- **Medical AI Community**: Open-source medical NLP tools and datasets

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the logs for error details

---

**⚠️ Medical Disclaimer**: This system is for educational and research purposes only. Always consult qualified healthcare professionals for medical decisions. The AI recommendations should not replace professional medical advice.