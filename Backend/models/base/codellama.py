from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables from .env file
load_dotenv()

# Get HuggingFace token from the environment variable
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Login to HuggingFace using the token
login(hf_token)

# Load CodeLlama tokenizer and model 
model_name = "codellama/CodeLlama-7b-Python-hf" 


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a code generation prompt
input_prompt = "Write a Python function to sort a list of numbers"

# Tokenize the input and generate code
inputs = tokenizer(input_prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=100)

# Decode and print the generated code
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n\nGenerated Code:\n")
print(generated_code)
