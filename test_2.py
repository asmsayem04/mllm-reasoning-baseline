"""
Experiment 2: Compositional Attribute Binding Failure
Tests multi-object spatial relationship and color-attribute binding.
"""
from PIL import Image, ImageDraw
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Generate synthetic stimulus (Red on left, Green on right)
raw_image = Image.new('RGB', (400, 200), color='white')
draw = ImageDraw.Draw(raw_image)
draw.rectangle([50, 50, 150, 150], fill="red")
draw.rectangle([250, 50, 350, 150], fill="green")

# Helper function to generate model responses
def ask_model(prompt):
    inputs = processor(raw_image, prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

# Define prompts
prompt_1 = "What is the color of the object on the left?"
prompt_2 = "Is the red object to the left or right of the green object?"

# Output results
print("=" * 70)
print("EXPERIMENT 2: COMPOSITIONAL ATTRIBUTE BINDING FAILURE")
print("=" * 70)

print("--- Test 2.1: Attribute Extraction ---")
print(f"Prompt: {prompt_1}\nModel Output: {ask_model(prompt_1)}\n")

print("--- Test 2.2: Relational Positioning ---")
print(f"Prompt: {prompt_2}\nModel Output: {ask_model(prompt_2)}")
print("=" * 70)