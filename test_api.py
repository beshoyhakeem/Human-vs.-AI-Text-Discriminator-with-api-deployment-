from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import uvicorn
import os

# Initialize FastAPI
app = FastAPI()

# Setup templates and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model loading
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "bert_ai_human_classifier")

print(f"Loading model from: {model_path}")
print("Model files:", os.listdir(model_path))

try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class TextRequest(BaseModel):
    text: str

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    return text.strip()

@app.post("/api/predict")
async def api_predict(request: TextRequest):
    """API endpoint for programmatic access"""
    inputs = tokenizer(
        clean_text(request.text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=192
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)

    return {
        "prediction": "AI" if prediction.item() else "Human",
        "confidence": confidence.item(),
        "probabilities": {
            "human": probs[0][0].item(),
            "ai": probs[0][1].item()
        }
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main UI page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_ui(
    request: Request,
    text: str = Form(...)
):
    """Handle form submissions from the UI"""
    result = await api_predict(TextRequest(text=text))
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "input_text": text
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)