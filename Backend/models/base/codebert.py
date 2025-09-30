from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a pre-trained model and tokenizer
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Encode input 
input_text = "Write a Python function to sort a list of numbers"  
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
outputs = model.generate(inputs["input_ids"], max_length=100)

# Decode output to get the code
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n\nGenerated Code:\n")
print(generated_code)
