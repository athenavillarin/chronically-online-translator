# Model loading + generate logic
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the trained model
model_dir = Path(__file__).parent / "fine_tuned_model"
model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

# Test examples (slang to standard)
test_examples = [
    "i'm finna hit up the cafeteria for some chow",
    "i'm lowkey drained",
    "that's cap bro",
    "no cap that was bussin",
]

print("Translation Results:\n")
for slang_text in test_examples:
    input_text = f"translate slang to standard: {slang_text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=128)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Input (slang):  {slang_text}")
    print(f"Output (standard): {translation}")
    print()