from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re

app = FastAPI(title="Citizen Grievance NLP API", version="1.0")

# --- Load models at startup ---
with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_dept = pickle.load(f)
with open('../models/dept_classifier.pkl', 'rb') as f:
    dept_model = pickle.load(f)
with open('../models/tfidf_sentiment.pkl', 'rb') as f:
    tfidf_sent = pickle.load(f)
with open('../models/sentiment_classifier.pkl', 'rb') as f:
    sent_model = pickle.load(f)

# --- Priority map ---
PRIORITY_MAP = {
    'Positive': 1,
    'Neutral': 2,
    'Negative': 3,
    'Critical/Urgent': 5
}

# --- Request schema ---
class GrievanceRequest(BaseModel):
    complaint: str

# --- Text cleaner ---
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "Grievance NLP API is running!"}

@app.post("/predict")
def predict(req: GrievanceRequest):
    cleaned = clean(req.complaint)

    # Department prediction
    dept_vec  = tfidf_dept.transform([cleaned])
    department = dept_model.predict(dept_vec)[0]

    # Sentiment prediction
    sent_vec  = tfidf_sent.transform([cleaned])
    sentiment = sent_model.predict(sent_vec)[0]
    priority  = PRIORITY_MAP.get(sentiment, 2)

    return {
        "complaint": req.complaint,
        "predicted_department": department,
        "sentiment": sentiment,
        "priority_score": priority
    }