# 🔧 LoRA Fine-Tuning Estimation & Setup

This project provides a starter notebook for estimating GPU memory usage for different fine-tuning strategies (Full Fine-Tuning, LoRA, and QLoRA), and demonstrates how to apply LoRA to a Hugging Face transformer model using the `peft` library.

---

## 📁 Contents

- `fine-tuning.ipynb`: Jupyter notebook containing:
  - VRAM estimation logic for various fine-tuning strategies.
  - LoRA setup example on a tiny GPT-2 model (`sshleifer/tiny-gpt2`).
  - Parameter statistics for base vs. trainable layers.

---

## 📦 Requirements

Install required packages:

```bash
pip -q install transformers peft bitsandbytes accelerate --upgrade --quiet
```
