# Fine-tuning a Language Model for Magento 2 REST API Question Answering on Apple Silicon

## Project Overview

This project focuses on fine-tuning a pre-trained language model to answer questions specifically about the Magento 2 REST API. The goal is to create a helpful assistant for developers working with Magento's API, runnable and trainable on a Mac with Apple Silicon (M-series chip) utilizing Metal Performance Shaders (MPS) for acceleration.

The project involves:
1.  Selecting a suitable pre-trained model.
2.  Creating a custom dataset of Magento 2 REST API questions and answers.
3.  Developing a training script using PyTorch and Hugging Face libraries (Transformers, PEFT).
4.  Fine-tuning the model using Low-Rank Adaptation (LoRA) for efficiency.
5.  Providing instructions for setup, training, and inference.

## Model and Dataset

*   **Model Used**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
    *   This is a 1.5 billion parameter model distilled from the larger DeepSeek-R1, showing strong reasoning capabilities, including for code. Its Qwen2.5 base architecture is well-suited for various NLP tasks.
*   **Dataset Used**: `magento_api_qna.jsonl`
    *   A custom-created dataset containing 50 question-answer pairs specifically about the Magento 2 REST API. The Q&A pairs were derived from the official Adobe Commerce (Magento) developer documentation.
    *   The process for creating this dataset is documented in `magento_api_dataset_creation.md`.

## Hardware Requirements

*   A Mac with an Apple Silicon chip (e.g., M1, M2, M3 series).
*   Sufficient RAM (16GB recommended, especially for the 1.5B parameter model, though LoRA helps reduce memory footprint during training).
*   macOS version that supports PyTorch with Metal (MPS).

## Setup Instructions

1.  **Clone the Repository**
    ```bash
    # This project is part of a larger repository.
    # Assuming you have cloned the main repository containing these files.
    # cd path/to/this/project_directory
    ```

2.  **Create a Virtual Environment**
    It's highly recommended to use a virtual environment to manage dependencies:
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the Virtual Environment**
    ```bash
    source venv/bin/activate
    ```
    *(On Windows, it would be `venv\Scripts\activate`)*

4.  **Install Dependencies**
    Install the necessary Python libraries using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This will install PyTorch, Transformers, PEFT, Datasets, Accelerate, and other required packages.

## Running the Training Script

Once the setup is complete, you can start fine-tuning the model using the `train_magento_model.py` script:

```bash
python train_magento_model.py
```

*   The script will use the MPS device if available on your Mac for hardware acceleration.
*   Training progress will be logged to the console.
*   The fine-tuned LoRA adapters will be saved to the `./magento_lora_adapters/` directory.

## Inference with the Fine-tuned Model

After training, the LoRA adapters can be loaded on top of the original pre-trained model for inference.

Here's a Python code snippet demonstrating how to do this:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_ADAPTERS_PATH = "./magento_lora_adapters/" # Path where your LoRA adapters are saved
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Load Tokenizer and Base Model ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32, # Use float16 on MPS for efficiency if supported
).to(DEVICE)

# --- Load LoRA Adapters ---
# Ensure the base model is loaded before applying adapters
model_with_adapters = PeftModel.from_pretrained(base_model, LORA_ADAPTERS_PATH)
model_with_adapters.eval() # Set to evaluation mode

print(f"Model with LoRA adapters loaded on {model_with_adapters.device}")

# --- Prepare Prompt for Inference ---
# The prompt format must match the one used during training.
def format_inference_prompt(question):
    return f"""Below is a question about the Magento 2 REST API. Provide a helpful and accurate answer.

### Question:
{question}

### Answer:""" # Note: The model will generate the text after "### Answer:"

# --- Example Inference ---
sample_question = "How do you get an admin token for Magento 2 REST API access?"
prompt = format_inference_prompt(sample_question)

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False).to(DEVICE)

print(f"\nGenerating answer for: {sample_question}")

with torch.no_grad(): # Disable gradient calculations for inference
    # outputs = model_with_adapters.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    # More robust generation arguments:
    outputs = model_with_adapters.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,  # Adjust as needed
        temperature=0.7,     # Controls randomness; lower is more deterministic
        top_p=0.9,           # Nucleus sampling
        do_sample=True,      # Enable sampling
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id # Important for some models
    )

generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# The output will contain the prompt, so we can extract just the answer part
# by finding where "### Answer:" ends in the generated text.
answer_start_tag = "### Answer:"
answer_index = generated_answer.find(answer_start_tag)
if answer_index != -1:
    final_answer = generated_answer[answer_index + len(answer_start_tag):].strip()
else:
    final_answer = "Could not parse the answer from model output." # Fallback

print(f"\nQuestion: {sample_question}")
print(f"Generated Answer: {final_answer}")

# Example with another question:
# sample_question_2 = "What is the purpose of bulk endpoints in the Magento 2 REST API?"
# prompt_2 = format_inference_prompt(sample_question_2)
# inputs_2 = tokenizer(prompt_2, return_tensors="pt", truncation=True, padding=False).to(DEVICE)
# with torch.no_grad():
#     outputs_2 = model_with_adapters.generate(**inputs_2, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
# generated_answer_2 = tokenizer.decode(outputs_2[0], skip_special_tokens=True)
# print(f"\nQuestion: {sample_question_2}")
# print(f"Generated Answer (raw): {generated_answer_2}")
```

## Output

*   The primary output of the training process is the set of LoRA adapters, saved in the `./magento_lora_adapters/` directory. This directory will contain files like `adapter_model.bin` (or `.safetensors`) and `adapter_config.json`.

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning (PEFT) technique. Instead of fine-tuning all the parameters of a large pre-trained model (which is computationally expensive and memory-intensive), LoRA freezes the original model weights and injects trainable rank decomposition matrices (adapters) into specific layers (typically attention layers) of the model.

**Why use LoRA for this project?**
*   **Efficiency**: Training only the adapter weights significantly reduces the number of trainable parameters (often by >90%). This makes fine-tuning large models feasible on consumer hardware like Macs with limited VRAM.
*   **Reduced Storage**: The fine-tuned LoRA adapters are much smaller than the original model (megabytes vs. gigabytes). This makes it easy to store, share, and switch between different fine-tuned tasks without duplicating the large base model.
*   **Comparable Performance**: For many tasks, LoRA can achieve performance comparable to full fine-tuning while being much more efficient.

This approach allows us to adapt the powerful `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model to our specific Magento 2 REST API Q&A task without requiring extensive computational resources.
