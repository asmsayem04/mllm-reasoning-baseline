"""
Experiment 1: Adversarial Geometry & Prompt Fragility
Tests zero-shot physical reasoning and response stability.
"""
import requests
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load stimulus (Penrose Triangle)
img_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Penrose-dreieck.svg/960px-Penrose-dreieck.svg.png'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
response = requests.get(img_url, headers=headers)
raw_image = Image.open(BytesIO(response.content)).convert('RGB')

# Helper function to generate model responses
def ask_model(prompt):
    inputs = processor(raw_image, prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

# Define prompts
prompt_0 = "What is the main geometric shape in this image?"
prompt_1 = "Is there anything physically impossible or logically wrong in this image?"
prompt_2 = "Is there anything physically impossible or logically wrong in this image? If Yes tell me the wrong thing you have find."

# Output results
print("=" * 70)
print("EXPERIMENT 1: ADVERSARIAL GEOMETRY & PROMPT FRAGILITY")
print("=" * 70)

print("--- Test 1.0: Basic Shape Detection ---")
print(f"Prompt: {prompt_0}\nModel Output: {ask_model(prompt_0)}\n")

print("--- Test 1.1: Binary Logical Query ---")
print(f"Prompt: {prompt_1}\nModel Output: {ask_model(prompt_1)}\n")

print("--- Test 1.2: Conditional Stress Test ---")
print(f"Prompt: {prompt_2}\nModel Output: {ask_model(prompt_2)}")
print("=" * 70)