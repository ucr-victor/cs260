from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Or your fine-tuned model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test function for the model
def test_model():
    test_questions = [
        "How do I change my router's password?",
        "Why is my internet connection unstable?",
        "How do I reset my router to factory settings?"
    ]
    
    for question in test_questions:
        inputs = tokenizer(question, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Question: {question}")
        print(f"Response: {response}\n")

# Run the test
test_model()
