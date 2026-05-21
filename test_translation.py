from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "backend/fine_tuned_model"

try:
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✓ Model loaded successfully")
    print("Enter Gen Z slang phrases to translate (type 'quit' to exit)\n")
    
    while True:
        phrase = input("Slang: ").strip()
        if phrase.lower() == 'quit':
            break
        
        input_text = f"translate slang to standard: {phrase}"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=128)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"English: {translation}\n")
        
except FileNotFoundError:
    print("❌ Model not found at", MODEL_PATH)
    print("Train the model first: python training/train.py")