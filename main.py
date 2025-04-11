from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import uvicorn
import os

# Initialize app
app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model loading
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "bert_ai_human_classifier")

print(f"Loading model from: {model_path}")
print("Model files:", os.listdir(model_path))

try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
    model_path,
    use_safetensors=True  # Explicitly enable safetensors
)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main UI page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text: str = Form(...)
):
    """Handle form submissions"""
    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=192
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)

    result = {
        "prediction": "AI" if prediction.item() else "Human",
        "confidence": f"{confidence.item() * 100:.2f}%",
        "probabilities": {
            "human": f"{probs[0][0].item() * 100:.2f}%",
            "ai": f"{probs[0][1].item() * 100:.2f}%"
        },
        "is_ai": prediction.item()
    }

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