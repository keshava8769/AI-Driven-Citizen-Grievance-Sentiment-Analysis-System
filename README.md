# Project 1: AI-Driven Citizen Grievance & Sentiment Analysis System
### Infotact ML Internship — Government & Public Sector Track

## 🧠 Overview
An end-to-end NLP pipeline that automatically:
- Routes citizen complaints to the correct government department
- Detects sentiment and urgency using fine-tuned BERT
- Serves predictions via a REST API built with FastAPI

## 🗂️ Project Structure
```
project1-grievance-nlp/
├── data/               # Dataset and EDA visualizations
├── notebooks/          # Week-wise Jupyter Notebooks
├── models/             # Saved model files (gitignored)
├── app/main.py         # FastAPI application
├── requirements.txt
└── README.md
```

## ⚙️ Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## ▶️ Run API
```bash
cd app
uvicorn main:app --reload
```

## 📡 API Endpoint
**POST** `/predict`
```json
Input:  { "complaint": "No water supply for 10 days" }
Output: {
  "predicted_department": "Water",
  "sentiment": "Critical/Urgent",
  "priority_score": 5,
  "priority_label": "Critical — Immediate Action Required"
}
```

## 📊 Model Performance
| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest (Department) | 94.5% | 0.9449 |
| BERT Fine-tuned (Sentiment) | 92.0% | 0.9195 |

## 🛠️ Tech Stack
Python · NLTK · Scikit-learn · HuggingFace Transformers · FastAPI · PyTorch

## 📅 Development Log
- Week 1: Data generation, text preprocessing, EDA
- Week 2: TF-IDF vectorization, department classification
- Week 3: SVM baseline → BERT fine-tuning for sentiment
- Week 4: Evaluation, FastAPI deployment, final documentation