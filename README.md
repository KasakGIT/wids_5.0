# Mini GPT from Scratch 🧠✨

This project is a **from-scratch implementation of a language model**, built step by step using **PyTorch**.

We start with a **Bigram Language Model** to understand the basics, and then gradually upgrade it into a **GPT-style Transformer model** by adding attention, feedforward layers, and positional embeddings.

---

## 📁 Project Structure


---

## 🚀 Step 1: Bigram Language Model (`bigram.py`)

The project begins with a **bigram model**, which predicts the **next character using only the current character**.

### Key ideas:
- Each character is mapped to an index
- A lookup table learns probabilities of next characters
- No context beyond one character

This model helps understand:
- Tokenization
- Embeddings
- Loss calculation
- Text generation loop

📉 **Limitation:**  
It has no memory of long-term context, so generated text is mostly random.

---

## 🚀 Step 2: GPT Model (`gpt.py`)

To overcome the limitations of the bigram model, we upgrade it into a **Transformer-based GPT model**.

The following components are added:

### ✅ Head (Single Self-Attention Head)
- Computes **query, key, and value**
- Uses masked self-attention
- Allows each token to attend to previous tokens

---

### ✅ MultiHeadAttention
- Runs **multiple attention heads in parallel**
- Captures different relationships between tokens
- Concatenates outputs of all heads

---

### ✅ FeedForward Network
- Two linear layers with a non-linearity
- Applied independently to each token
- Helps the model learn complex transformations

---

### ✅ Transformer Block
Combines:
- Multi-head attention
- Feedforward network
- Residual connections
- Layer normalization

This block is stacked multiple times to build depth.

---

### ✅ Positional Embeddings
- Adds information about **token order**
- Necessary because attention alone is position-agnostic
- Combined with token embeddings

---

## 🧠 Final Model Architecture

---

## 🏋️ Training

- Model is trained on `input.txt`
- Uses cross-entropy loss
- Optimized using AdamW
- Trained to predict the next character

---

## ✨ Text Generation

After training, the model can:
- Generate text autoregressively
- Use learned attention patterns
- Produce more coherent output than the bigram model

---

## 📌 Key Learnings

- Why bigram models are limited
- How self-attention works
- Why positional embeddings are required
- How GPT models generate text
- How multiple simple blocks combine into a powerful model

---

## 📚 Inspiration

Inspired by:
- Andrej Karpathy’s nanoGPT
- Transformer architecture from “Attention Is All You Need”

---

## 🔧 Requirements

- Python 3.x
- PyTorch

---

## ▶️ How to Run

```bash
python bigram.py
python gpt.py

