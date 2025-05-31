import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration ---
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH = "magento_api_qna.jsonl"
OUTPUT_DIR = "./magento_lora_adapters"
LORA_R = 16  # Rank of the LoRA matrices
LORA_ALPHA = 32  # Alpha parameter for LoRA scaling
LORA_DROPOUT = 0.05
# Qwen2 specific target modules or common ones. Might need inspection.
# Based on Qwen2 architecture, typical names are:
# For self-attention: q_proj, k_proj, v_proj, o_proj
# For MLP: gate_proj, up_proj, down_proj
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

NUM_EPOCHS = 3
BATCH_SIZE = 1 # Keep low for Mac MPS memory
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512 # Max sequence length for tokenization

# --- Device Setup ---
# Explicitly set to MPS if available, otherwise CPU.
# CUDA is unlikely on a Mac, but good practice to check.
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda" # Should not happen on Mac
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# --- Load and Preprocess Dataset ---
def create_prompt(example):
    # Simple instruction-following format
    prompt_template = f"""Below is a question about the Magento 2 REST API. Provide a helpful and accurate answer.

### Question:
{example['question']}

### Answer:
{example['answer']}"""
    return {"text": prompt_template}

# Load dataset
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Apply prompt formatting
dataset = raw_dataset.map(create_prompt)

# --- Load Tokenizer and Model ---
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Set padding token if not already set (common for Qwen/LLaMA based models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Default for Causal LM

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length", # Pad to max_length
        max_length=MAX_SEQ_LENGTH,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["question", "answer", "text"])
print(f"Sample tokenized input: {tokenized_dataset[0]['input_ids'][:20]}") # Print some token IDs

# Load model
# For MPS, bitsandbytes quantization (load_in_8bit/4bit) is generally not supported.
# We'll load in native precision or float16 if memory is an issue and it's supported.
# The model card says BF16, which MPS might handle. If not, it will default to float32 or float16.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    # torch_dtype=torch.bfloat16, # Specify if MPS has good BF16 support, else auto
).to(DEVICE)

print(f"Model loaded on {model.device}")

# --- PEFT (LoRA) Setup ---
# Although we are not using k-bit training like load_in_8bit,
# prepare_model_for_kbit_training can still be useful for preparing the model
# for PEFT, such as casting LayerNorm and other layers to float32 for stability.
# model = prepare_model_for_kbit_training(model) # Commenting out for now, test without first for MPS

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",  # Typically 'none' for LoRA
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Training ---
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="epoch", # Save checkpoints at the end of each epoch
    # report_to="tensorboard", # Or "wandb" or "none"
    report_to="none", # Keep it simple for now
    fp16=False,  # Set to True if using mixed precision and device supports it. MPS has some fp16 support.
                 # BF16 is generally better if supported (model native precision).
                 # For MPS, explicit mixed precision needs careful handling.
    # mps_specific_training_args if any, often handled by `accelerate` or device placement
    # Forcing device, though Trainer should pick it up from model.device
    # no_cuda=True, # If you want to be very explicit about not using CUDA
)

# Data collator for language modeling.
# It will handle padding and also create labels for us (input_ids will be shifted for labels).
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# --- Save LoRA Adapters ---
print(f"Saving LoRA adapters to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
# To save tokenizer as well, if needed for inference later
# tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete and adapters saved.")

# To load the LoRA adapters later for inference:
# from peft import PeftModel
# base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
# model_with_adapters = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
# model_with_adapters.to(DEVICE)
# model_with_adapters.eval()
# print("LoRA adapters loaded for inference.")
#
# Example inference (pseudo-code):
# prompt = "..." # Formatted prompt
# inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
# outputs = model_with_adapters.generate(**inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#
# For MPS, ensure all tensors are on the 'mps' device.
# If issues arise, consider:
# 1. Smaller batch size (already 1)
# 2. Gradient accumulation (training_args.gradient_accumulation_steps)
# 3. Simpler LoRA config (fewer target_modules, smaller R)
# 4. Ensure torch version is recent and supports MPS well for all ops.
# 5. If `prepare_model_for_kbit_training` causes issues on MPS, it can be omitted if not using quantization.
#    It mainly helps with datatype casting for stability with quantized models.
# 6. Check `torch_dtype` for model loading; MPS might prefer `torch.float32` or `torch.float16`
#    over `torch.bfloat16` if full support for the latter is lacking.
#    The model card states BF16, so ideally that works. Transformers might handle conversion.
# 7. `fp16=True` in TrainingArguments can be tried, but full mixed-precision support on MPS
#    has historically been less mature than CUDA's.
#
# If `bitsandbytes` was to be used for quantization (e.g. `load_in_8bit=True`):
# - It generally requires CUDA. There are experimental/limited versions for other platforms,
#   but full functionality for training/inference quantization might not be there for MPS.
# - If it were compatible, `prepare_model_for_kbit_training(model)` would be more important.
# - This script avoids `load_in_Xbit` to maximize MPS compatibility for now.
#
# Target modules for Qwen2 based on common Hugging Face implementations:
# model.model.layers[0].self_attn has q_proj, k_proj, v_proj, o_proj
# model.model.layers[0].mlp has gate_proj, up_proj, down_proj
# This seems correct for LORA_TARGET_MODULES.
#
# A note on `tokenizer.padding_side = "right"`:
# For decoder-only models (like GPT, LLaMA, Qwen), "left" padding is often recommended
# during training to allow for batched generation without issues. However, "right" padding
# is simpler and often the default; it works fine for training if you're not doing
# batched generation immediately after with the same setup.
# Since `DataCollatorForLanguageModeling` handles labels by shifting input_ids,
# the padding side's main impact is on how inputs are structured.
# For consistency and common practice with Causal LMs, "left" can be safer if issues arise.
# Let's stick to "right" for now as it's often a default in examples.
# If `eos_token` is used for padding, ensure attention mask correctly ignores it.
# `DataCollatorForLanguageModeling` should handle this.
# The model card for DeepSeek-R1-Distill-Qwen-1.5B does not specify a chat template directly,
# but mentions it can be used like Qwen or Llama. The chosen instruction format is generic.
# The tokenizer for Qwen models usually handles special tokens like <|im_start|> and <|im_end|>.
# My custom prompt does not use these, which is fine for fine-tuning, as the model will adapt.
# If I were to use its pre-trained chat capabilities, I'd need the exact template.
# For fine-tuning on a new task/format, this is less critical as long as the format is consistent.
#
# The tokenizer `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` has `eos_token = "<|endoftext|>"`,
# `pad_token = "<|endoftext|>"` and `bos_token = "<|endoftext|>"`. This is fine.
# Using `eos_token` as `pad_token` is a common strategy. Max sequence length will handle padding.
#
# Final check on MPS compatibility:
# - `torch.bfloat16`: Support on MPS can be limited. The model might be loaded as FP32 or FP16 by default if BF16 is not well-supported by `to(DEVICE)`.
#   The `AutoModelForCausalLM.from_pretrained` call without explicit `torch_dtype` will use the model's config,
#   and then `.to(DEVICE)` will move it. This should be okay.
# - `fp16=False` in `TrainingArguments` is safer to start. If memory is an issue and performance is slow,
#   can try `fp16=True`, but it might require `torch.mps.empty_cache()` and careful checks.
#
# `prepare_model_for_kbit_training` is primarily for models loaded with k-bit quantization (e.g., 8-bit, 4-bit via bitsandbytes).
# Since I am not loading with k-bit quantization to maintain MPS compatibility, this call might not be strictly necessary
# and could potentially cause issues if it tries to perform operations not well-supported on MPS.
# I've commented it out for the initial attempt. If there are stability issues (e.g., with LayerNorms),
# it could be re-introduced, but its main benefit is for quantized models.

print("Script definition complete. Ready to be saved.")
