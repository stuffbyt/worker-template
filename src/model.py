import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Model:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        print("Loading model...")
        # Get model name from environment variable or use default
        model_name = os.environ.get("MODEL_NAME", "gpt2")
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"Model loaded successfully on {self.device}")
        
    def generate(self, prompt, max_length=None, temperature=None):
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Get defaults from environment variables if not provided
        if max_length is None:
            max_length = int(os.environ.get("DEFAULT_MAX_LENGTH", 100))
        if temperature is None:
            temperature = float(os.environ.get("DEFAULT_TEMPERATURE", 0.7))
            
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=temperature > 0
            )
            
        # Decode and return the generated text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
