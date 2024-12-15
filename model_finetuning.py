from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Sample training data with question-answer pairs
train_data = [
    {"input": "How do I reset my router?", "output": "Press and hold the reset button for 10 seconds."},
    {"input": "What is WPA2?", "output": "WPA2 is a security protocol for wireless networks."},
    # Add more Q&A pairs here
]

# Tokenize the training data
train_encodings = tokenizer([item['input'] for item in train_data], truncation=True, padding=True)
train_labels = tokenizer([item['output'] for item in train_data], truncation=True, padding=True)

# Create a custom dataset for training
class RouterDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'labels': self.labels['input_ids'][idx]
        }

    def __len__(self):
        return len(self.encodings['input_ids'])

# Prepare the dataset
train_dataset = RouterDataset(train_encodings, train_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=8,   
    warmup_steps=500,               
    weight_decay=0.01,               
    logging_dir='./logs',            
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                        
    args=training_args,                 
    train_dataset=train_dataset,         
)

# Fine-tune the model
trainer.train()
