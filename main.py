from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from typing import Dict, Any

class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis")
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")


@app.get("/")
def root():
    return {"FastApi service started!"}


@app.get("/{text}")
def get_params(text: str):
    return classifier(text)


@app.post("/predict/")
def analyze_text(item: Item) -> Dict[str, Any]:

    sentiment_result = classifier(item.text)[0]
    emotion_result = emotion_classifier(item.text)[0]

    return {
        "text": item.text,
        "sentiment": {
            "label": sentiment_result["label"],
            "score": sentiment_result["score"]
        },
        "emotion": {
            "label": emotion_result["label"],
            "score": emotion_result["score"]
        },
        "text_length": len(item.text),
        "analysis": "completed"
    }


@app.get("/health/")
def health_check() -> Dict[str, str]:
    return {"status": "healthy"}
