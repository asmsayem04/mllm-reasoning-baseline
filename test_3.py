"""
Experiment 3: Visual Numeracy & Attention Fragmentation
Evaluates quantitative reasoning and precise object enumeration.
"""
from PIL import Image, ImageDraw
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Generate synthetic stimulus (Exactly 7 blue circles)
raw_image = Image.new('RGB', (400, 400), color='white')
draw = ImageDraw.Draw(raw_image)
coordinates = [(50,50), (150,80), (250,50), (100, 200), (300, 220), (150, 300), (250, 310)]
for (x, y) in coordinates:
    draw.ellipse([x, y, x+40, y+40], fill="blue")

# Helper function to generate model responses
def ask_model(prompt):
    inputs = processor(raw_image, prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

# Define prompt
prompt_1 = "How many blue circles are in this image?"

# Output results
print("=" * 70)
print("EXPERIMENT 3: VISUAL NUMERACY & ENUMERATION FAILURE")
print("=" * 70)

print("--- Test 3.1: Quantitative Reasoning ---")
print(f"Prompt: {prompt_1}")
print(f"Model Output: {ask_model(prompt_1)}")
print(f"Expected Output: 7")
print("=" * 70)