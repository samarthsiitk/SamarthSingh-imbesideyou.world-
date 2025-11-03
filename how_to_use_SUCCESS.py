# Your Code-to-Research Pipeline AI Agent - SUCCESS!

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load your successfully trained model
base_model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load your trained LoRA adapter
model = PeftModel.from_pretrained(base_model, "/content/code-research-agent-lora/successful_lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("/content/code-research-agent-lora/tokenizer")

def generate_code(instruction):
    """Generate Python code from research instructions"""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.3
        )
    
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()

# Example usage - Your model now works!
code = generate_code("Write a function to preprocess image data for machine learning")
print(code)

# Your model will generate proper Python code like:
# def preprocess_image_data(image_path):
#     import cv2
#     import numpy as np
#     # ... proper implementation
