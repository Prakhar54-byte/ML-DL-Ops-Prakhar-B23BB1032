from transformers import MarianMTModel, MarianTokenizer
from striprtf.striprtf import rtf_to_text
import torch

model_name = "Helsinki-NLP/opus-mt-bn-en"

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Read RTF file
with open("input.rtf", "r", encoding="utf-8") as f:
    rtf_text = f.read()

plain_text = rtf_to_text(rtf_text)
lines = [l.strip() for l in plain_text.split("\n") if l.strip() != ""]

outputs = []
batch_size = 16

print(f"Total lines to translate: {len(lines)}")

for i in range(0, len(lines), batch_size):
    batch = lines[i:i+batch_size]
    print(f"Translating lines {i+1} to {min(i+batch_size, len(lines))}...")
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs)
    texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    outputs.extend(texts)

with open("output.txt", "w", encoding="utf-8") as f:
    for line in outputs:
        f.write(line + "\n")

print("Translation completed!")