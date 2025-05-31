# Pre-trained Model Selection for Code-Related Question Answering

## Objective

To research and select a suitable pre-trained model for code-related question answering that can be fine-tuned on a Mac with Metal support, prioritizing models available through Hugging Face Transformers.

## Selected Model

**Model ID:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

**Link:** [https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

## Reasons for Selection

1.  **Strong Code Capabilities:**
    *   The DeepSeek-R1 series, of which this model is a part, is explicitly designed and evaluated for reasoning tasks that include code. The model card highlights performance on benchmarks like LiveCodeBench and Codeforces.
    *   It is distilled from larger DeepSeek-R1 models, which have demonstrated strong performance in math, code, and general reasoning. This distillation process aims to transfer reasoning capabilities to a smaller model.

2.  **Manageable Size for Mac Fine-tuning:**
    *   The model has 1.78 billion parameters and is available in BF16 precision. While not a "small" model, it's considerably more manageable than larger models (e.g., 7B+ parameters).
    *   Fine-tuning on a Mac with Metal (using PyTorch's MPS backend) should be feasible, particularly when employing parameter-efficient fine-tuning (PEFT) techniques like LoRA (Low-Rank Adaptation) and potentially quantization. This helps manage memory requirements (weights, optimizer states, activations).

3.  **Hugging Face Transformers Compatibility:**
    *   The model is available on the Hugging Face Hub and is described as usable in the same manner as standard Qwen or Llama models. This ensures compatibility with the Hugging Face Transformers library for straightforward loading, fine-tuning, and inference.
    *   It uses the Qwen2.5 architecture as its base (`Qwen/Qwen2.5-Math-1.5B`), which is supported in Transformers.

4.  **Good Balance of Performance and Size:**
    *   The model shows respectable performance on various benchmarks, including those related to code, for its size class. This makes it a good compromise between the capabilities of very large models and the resource constraints of local Mac hardware.

5.  **Active Development and Permissive License:**
    *   The model is part of a recent series from DeepSeek-AI, indicating active development and support.
    *   It is released under an MIT License, which is permissive for research, modification, and commercial use.

## Alternatives Considered

*   **`CodeLifeCL/distilbert-base-uncased-finetuned-squad-d5716d28`**:
    *   **Pros:** Much smaller (DistilBERT), already fine-tuned for QA (SQuAD).
    *   **Cons:** Not specifically pre-trained or fine-tuned on code. Would require significant code-specific fine-tuning.
*   **`eagle0504/openai-gsm8k-codealpaca-20k-enhanced-deepseek-r1-distill-qwen-1.5b`**:
    *   **Pros:** Uses CodeAlpaca dataset, distilled.
    *   **Cons:** Seemed more focused on math reasoning (GSM8K). The selected DeepSeek model is a newer, more direct offering from the creators of the underlying technology and has a broader focus on reasoning including code.
*   **Larger Models (e.g., CodeLlama 7B+, other Gemma/Llama variants):**
    *   **Pros:** Potentially higher performance.
    *   **Cons:** Likely too large for effective fine-tuning on typical Mac hardware even with Metal, due to memory and computational constraints.

## Next Steps (Post-Selection)

The next step would be to acquire a suitable code-related question-answering dataset and fine-tune `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on this dataset using a Mac with Metal support, leveraging PEFT techniques.
