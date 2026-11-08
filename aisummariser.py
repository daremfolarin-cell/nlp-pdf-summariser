!pip install -q transformers torch pypdf accelerate
!pip install pypdf

import torch
from transformers import pipeline
from pypdf import PdfReader
from google.colab import files

print("Downloading and initializing Qwen2.5-0.5B-Instruct (No tokens required)...")

print("Downloading and initializing BART-Large-CNN...")
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

print("AI Model is fully loaded and ready to use!")

uploaded = files.upload()

filename = next(iter(uploaded))

print("You uploaded:",filename)

reader = PdfReader(filename)

texts= ""

print(f"The file {filename} contains {len(reader.pages)} page(s).")

for page in reader.pages:
  texts += page.extract_text() + "\n"

short_text = texts[:5000]

prompt = f"""
Summarize the following text.
Include the important ideas and details.

TEXT:
{short_text}

SUMMARY:
"""

result = generator(prompt, max_new_tokens=300)

print(result[0]["generated_text"])
