import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Llama model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Function to generate responses from the model
def generate_response(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate the response from the model
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    
    # Decode the generated output to human-readable text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example: Query router setup question
input_text = "How do I set up my router?"
response = generate_response(input_text)
print(response)
