from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizerFast.from_pretrained("./saved_model")
model.eval()

class TextInput(BaseModel):
    text: str

sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.post("/predict")
async def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = int(torch.argmax(outputs.logits, dim=-1))
    return {"sentiment": sentiment_mapping[predicted_class]}
