from model import load_model, predict_text  # Updated import path

# 1. First load the model and tokenizer
print("Loading model...")
model, tokenizer = load_model()
print("Model loaded successfully!")

# 2. Sample texts to test
samples = [
    "The quick brown fox jumps over the lazy dog.",  # Human
    "The utilization of advanced neural network architectures...",  # AI
    # Add your other test cases here
]

# 3. Make predictions
for i, text in enumerate(samples, 1):
    print(f"\nTest Case #{i}:")
    print(f"Text: {text[:60]}...")
    
    try:
        result = predict_text(text, model, tokenizer)
        print(f"Prediction: {'AI-generated' if result['prediction'] else 'Human-written'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities - Human: {result['probabilities']['human']:.2%}, AI: {result['probabilities']['ai']:.2%}")
    except Exception as e:
        print(f"Error predicting: {str(e)}")
    
    print("-" * 80)