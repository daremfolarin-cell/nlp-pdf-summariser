AI-Powered PDF Summarizer
An automated, lightweight Python script designed for Google Colab that extracts text from uploaded PDF files and generates concise summaries using the open-source Qwen2.5-0.5B-Instruct Large Language Model (LLM).

Objective
Reading through long documents, reports, or research papers is time-consuming. This project provides a private, automated notebook script to upload any PDF file, programmatically extract its text contents, and instantly receive an AI-generated summary without needing paid API keys or external tokens.

Technologies & Components
Python: The core programming language orchestrating the pipeline.
Hugging Face Transformers: Downloads and runs the open-source LLM pipeline locally within the environment.
Qwen2.5-0.5B-Instruct: A lightweight, highly efficient 500-million parameter language model optimized for instructions and text summarization.
PyPDF: A pure-Python PDF library utilized to parse and extract raw text from uploaded document pages.
PyTorch & Accelerate: Backend deep learning frameworks used for tensor hardware acceleration.
Google Colab Files API: Provides an interactive user interface for uploading local files into the cloud runtime.

How It Works
Environment Setup: The script silently installs required libraries (`transformers`, `pypdf`, `accelerate`) into the workspace.
Model Loading: The Hugging Face pipeline fetches the `Qwen2.5-0.5B-Instruct` model weights directly into the environment's memory.
File Ingestion: The user is prompted with an upload button to select a local PDF file.
Text Extraction: The script loops through every single page of the PDF sequentially, extracting raw text and chaining it into a single string.
Context Windowing: The text is safely sliced to the first 5,000 characters to prevent exceeding the memory window of the model.
Inference: A structured prompt instructs the AI model to summarize the extracted data, returning up to 300 new tokens of refined summary text directly to the console.
 
How to Run or Reproduce
Open a new notebook at Google Colab.
Change the runtime type to use a GPU accelerator for maximum speed (Runtime > Change runtime type > T4 GPU).
Paste the following Python script into a code cell:
```python
# 1. Install required packages quietly
!pip install -q transformers torch pypdf accelerate

# 2. Import necessary libraries
import torch
from transformers import pipeline
from pypdf import PdfReader
from google.colab import files

# 3. Initialize the model pipeline
print("Downloading and initializing Qwen2.5-0.5B-Instruct...")
generator = pipeline(
    "text-generation", 
    model="Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16, # Uses less memory in Colab
    device_map="auto"          # Automatically uses GPU if available
)
print("AI Model is fully loaded and ready to use!\n")

# 4. Upload and read the PDF
uploaded = files.upload()
filename = next(iter(uploaded))
print(f"\nYou uploaded: {filename}")

reader = PdfReader(filename)
texts = ""
print(f"The file {filename} contains {len(reader.pages)} page(s).")

for page in reader.pages:
    text_content = page.extract_text()
    if text_content: # Ensure page has extractable text
        texts += text_content + "\n"

# 5. Slice text and generate the summary
short_text = texts[:5000]

prompt = f"""Summarize the following text. Include the important ideas and details.

TEXT: 
{short_text}

SUMMARY:"""

print("\nGenerating summary...")
result = generator(prompt, max_new_tokens=300, clean_up_tokenization_spaces=True)
print("\n" + result[0]["generated_text"])
```
Execute the cell (`Ctrl + Enter` or click the Play button).
Click Choose Files when prompted, select a local PDF, and wait for the model to automatically output the final text summary.

Limitations & Future Improvements
The 5,000-Character Ceiling: The script currently slices text at `[:5000]`. If you upload a lengthy document, it will only summarize the first few pages.
Improvement: Implement a "Chunking & Stuffing" (MapReduce) mechanism to break long texts into smaller chunks, summarize each chunk, and combine them for a final overview.
Scanned Image Formats: `PyPDF` only extracts native digital text. If a PDF consists of scanned images, the script returns an empty output.
Improvement: Integrate an Optical Character Recognition (OCR) engine like `pytesseract` to read image-based documents.
Model Size Restrictions: The 0.5B parameter model is chosen for speed and memory safety, but it may miss deep contextual nuances compared to larger variants.
Improvement: Scale up the pipeline to use `Qwen2.5-3B-Instruct` or `Qwen2.5-7B-Instruct` if running on high-RAM premium Colab GPU tiers.
