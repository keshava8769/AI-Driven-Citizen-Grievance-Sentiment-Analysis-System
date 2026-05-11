# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Citizen Grievance NLP API",
    description="Automatically routes citizen complaints to the correct department and scores urgency.",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# Load All Models at Startup
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Department models
with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_dept = pickle.load(f)
with open('../models/dept_classifier.pkl', 'rb') as f:
    dept_model = pickle.load(f)

# BERT sentiment model
tokenizer   = BertTokenizer.from_pretrained('../models/bert_sentiment/')
bert_model  = BertForSequenceClassification.from_pretrained('../models/bert_sentiment/')
bert_model.to(device)
bert_model.eval()

# Label encoder
with open('../models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# ─────────────────────────────────────────────
# Priority Score Mapping
# ─────────────────────────────────────────────
PRIORITY_MAP = {
    'Positive':        1,
    'Neutral':         2,
    'Negative':        3,
    'Critical/Urgent': 5
}

PRIORITY_LABEL = {
    1: "Low",
    2: "Medium",
    3: "High",
    5: "Critical — Immediate Action Required"
}

# ─────────────────────────────────────────────
# Request & Response Schemas
# ─────────────────────────────────────────────
class GrievanceRequest(BaseModel):
    complaint: str

    class Config:
        json_schema_extra = {
            "example": {
                "complaint": "There is no water supply in our area since 10 days. Children are falling sick."
            }
        }

class GrievanceResponse(BaseModel):
    complaint:            str
    predicted_department: str
    sentiment:            str
    priority_score:       int
    priority_label:       str

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def predict_department(clean: str) -> str:
    vec  = tfidf_dept.transform([clean])
    return dept_model.predict(vec)[0]

def predict_sentiment(clean: str) -> str:
    enc = tokenizer(
        [clean],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    input_ids      = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        output = bert_model(input_ids=input_ids, attention_mask=attention_mask)

    pred_idx = torch.argmax(output.logits, dim=1).item()
    return le.inverse_transform([pred_idx])[0]

# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Citizen Grievance NLP API is live!",
        "docs":    "Visit /docs for interactive API documentation"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict", response_model=GrievanceResponse)
def predict(req: GrievanceRequest):
    if not req.complaint.strip():
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")

    cleaned    = clean_text(req.complaint)
    department = predict_department(cleaned)
    sentiment  = predict_sentiment(cleaned)
    score      = PRIORITY_MAP.get(sentiment, 2)
    label      = PRIORITY_LABEL.get(score, "Medium")

    return GrievanceResponse(
        complaint=req.complaint,
        predicted_department=department,
        sentiment=sentiment,
        priority_score=score,
        priority_label=label
    )