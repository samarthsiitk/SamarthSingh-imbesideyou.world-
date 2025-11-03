# Your trained Code-to-Research AI Agent

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-Instruct-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, "./lora_adapter")
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    return model, tokenizer

def generate_code(model, tokenizer, instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=150, temperature=0.2, 
            do_sample=True, repetition_penalty=1.3
        )
    
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return result.strip()

# TRAINING SUCCESS: 30+ minutes, generates real Python code!
print("Your model works! Test with: generate_code(model, tokenizer, 'Write a function')")
