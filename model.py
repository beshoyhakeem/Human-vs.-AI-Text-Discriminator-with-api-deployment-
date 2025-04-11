from transformers import BertTokenizer, BertForSequenceClassification
import torch

class TextClassifier:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def clean_text_simple(self, text):
        """Basic text cleaning"""
        return text.strip()

    def predict(self, text):
        cleaned_text = self.clean_text_simple(text)
        
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=192
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

        return {
            "prediction": prediction.item(),
            "confidence": confidence.item(),
            "probabilities": {
                "human": probs[0][0].item(),
                "ai": probs[0][1].item()
            },
            "text": text
        }