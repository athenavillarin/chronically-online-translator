# Chronically Online Translator
**Context-Aware Neural Style Transfer for Internet Vernacular and Digital Slang Normalization**

An alternative learning assessment for **CCS 249: Natural Language Processing** that translates internet slang and Gen Z vernacular into professional, standard English using a fine-tuned neural language model.

---

## Project Overview

The Chronically Online Translator is a neural machine translation application designed to normalize digital slang and internet vernacular into formal, professional English. This project addresses the growing need for automatic text normalization in digital communication, where informal language, abbreviations, and slang are prevalent.

**Key Features:**
- High-accuracy slang-to-standard English translation
- Real-time inference with sub-second latency
- Clean, intuitive web interface
- Robust evaluation metrics (BLEU, ROUGE, cosine similarity)
- Fine-tuned Flan-T5 model for domain-specific performance

---

## Tech Stack

### Model & ML Framework
- **Base Model:** Flan-T5-Small (80M parameters)
- **Fine-tuning Framework:** Hugging Face Transformers
- **Deep Learning:** PyTorch
- **Training Acceleration:** Accelerate library

### Backend
- **Framework:** Flask with CORS support
- **Language:** Python 3.8+
- **Key Dependencies:**
  - `transformers` — Pre-trained language models
  - `torch` — PyTorch deep learning framework
  - `flask` — Web server
  - `scikit-learn` — Data processing utilities
  - `datasets` — HuggingFace datasets library

### Frontend
- **HTML5, CSS3, JavaScript**
- **Responsive design** for desktop and mobile

### Data & Evaluation
- **Data Format:** CSV (train/test splits)
- **Evaluation Metrics:** 
  - BLEU score
  - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
  - Cosine similarity (semantic similarity)

---

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- ~2GB of free disk space (for model weights)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd chronically-online-translator
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 4: Download Model Weights
The fine-tuned model weights are not included to avoid repository bloating. If you need to download or verify:
```bash
# Download the model weights from Google Drive:
https://drive.google.com/drive/folders/1RzhvJF7Y-U7UO_NRajVYlxetXcYCm6Vz?usp=sharing

# Place the contents into backend/fine_tuned_model/
```

---

## How to Run

### Starting the Application

#### Local Development
```bash
# Navigate to backend directory
cd backend

# Run Flask development server
python app.py
```

The application will start at `http://localhost:5000`

#### Using Environment Variables
```bash
# Set custom CORS origins (optional)
set CORS_ORIGINS="http://localhost:3000,http://yourdomain.com"
python app.py
```

### Using the API

#### Web Interface
1. Open `http://localhost:5000` in your browser
2. Enter slang text in the input field
3. Click "Translate" to get standard English output

#### REST API Endpoint

**POST** `/api/translate`

**Request:**
```json
{
  "text": "yo fr fr that's cap ngl"
}
```

**Response:**
```json
{
  "input": "yo fr fr that's cap ngl",
  "output": "truly, that is not truthful, honestly",
  "success": true
}
```

**Error Response:**
```json
{
  "error": "No text provided",
  "success": false
}
```

#### Example with cURL
```bash
curl -X POST http://localhost:5000/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"thats lowkey fire tho"}'
```

---

## Evaluation Results

The fine-tuned model demonstrates strong performance across multiple evaluation metrics:

### BLEU & ROUGE Scores
- **BLEU Score:** 0.6879 (68.79%)
- **ROUGE-1:** 0.8509 (85.09%)
- **ROUGE-2:** 0.7479 (74.79%)
- **ROUGE-L:** 0.8439 (84.39%)

### Semantic Similarity
- **Average Cosine Similarity:** 0.6009 (60.09%)
- **Range:** 0.23 - 0.89
- Indicates strong semantic preservation during translation

### Interpretation
- **BLEU/ROUGE scores** measure translation accuracy and overlap with reference translations
- **Cosine similarity** measures semantic preservation, ensuring the translated text maintains the original meaning
- High ROUGE scores indicate the model produces outputs very similar to expected translations
- Cosine similarity scores suggest the model effectively captures semantic meaning across slang-to-standard conversions

---

## Project Structure

```
chronically-online-translator/
├── backend/
│   ├── app.py                 # Flask application
│   ├── model.py              # Translation service
│   ├── evaluator.py          # Evaluation metrics
│   ├── requirements.txt       # Python dependencies
│   └── fine_tuned_model/      # Pre-trained model weights
├── frontend/
│   ├── index.html            # Web UI
│   ├── script.js             # Frontend logic
│   └── style.css             # Styling
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Train/test splits
│   └── prepare_data.py       # Process data
├── training/
│   └── train.py              # Model fine-tuning script
├── notebooks/
│   └── training_colab.ipynb   # Google Colab training notebook
└── README.md                 # This file
```

---

## Known Limitations

### Model Limitations
1. **Domain Coverage:** Model trained primarily on Gen Z internet slang. Performance may degrade on:
   - Regional dialects or non-English slang
   - Newly emerged slang terms not present in training data
   - Professional jargon or technical abbreviations

2. **Context Understanding:** The model operates on individual messages without long-range conversational context
   - May not handle multi-turn conversations optimally
   - Lacks ability to reference previous messages

3. **Homonym Handling:** Ambiguous slang terms may be translated inconsistently
   - Example: "hit" can mean "to strike" or "attractive/successful"
   - Model uses statistical probability, not true disambiguation

### Data & Performance Limitations
1. **Semantic Loss:** Some nuance in casual tone is necessarily lost when translating to formal English
2. **Rare Slang:** Out-of-vocabulary or very new slang terms may result in literal or incorrect translations
3. **Inference Speed:** First inference loads the model (~2-3 seconds); subsequent requests are fast

### Current Constraints
- **Model Size:** 80M parameters (suitable for CPU/small GPU inference)
- **Latency:** ~1-2 seconds per request (including model loading)
- **No Fine-grained Control:** No parameters to adjust formality level or translation style
- **Limited Error Handling:** May fail on extremely long inputs (>512 tokens)

### Future Improvements
- Expand training dataset to include more recent slang
- Implement conversational context tracking
- Add adjustable formality levels
- Deploy on edge devices for faster inference
- Multi-language support

---

### Training Data
 - Dataset Source: GenZ Slang Pairs (Programmer-RD-AI/genz-slang-pairs-1k)
 - Source: https://www.kaggle.com/datasets/programmerrdai/genz-slang-pairs-1k
 - Total pairs: ~1,000 sentence pairs, split 85/15 into train/test sets.
   
**Last Updated:** May 2026
