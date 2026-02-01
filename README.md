
---

## Step 1: Bigram Language Model (`bigram.py`)

The Bigram Language Model predicts the next character using only the current character.

### Key Concepts
- Character-level tokenization  
- Mapping characters to indices  
- Embedding lookup tables  
- Cross-entropy loss computation  
- Autoregressive text generation  

### Limitation
The model has no memory beyond one character, resulting in largely incoherent text generation.

---

## Step 2: GPT Model (`gpt.py`)

To overcome the limitations of the bigram model, a Transformer-based GPT model is implemented.

### Components

#### Self-Attention Head
- Computes Query, Key, and Value vectors  
- Uses causal masking to prevent access to future tokens  
- Enables context-aware token representations  

#### Multi-Head Attention
- Multiple attention heads operate in parallel  
- Captures diverse token relationships  
- Concatenates outputs of all heads  

#### Feedforward Network
- Two fully connected layers with non-linear activation  
- Applied independently to each token  

#### Transformer Block
- Multi-head self-attention  
- Feedforward network  
- Residual connections  
- Layer normalization  

#### Positional Embeddings
- Encode token position information  
- Added to token embeddings to preserve sequence order  

---

## Final Model Architecture

The final model consists of token embeddings, positional embeddings, stacked Transformer blocks, and a linear output projection layer.

---

## Training

- Dataset: `input.txt`  
- Objective: Next-character prediction  
- Loss function: Cross-Entropy Loss  
- Optimizer: AdamW  

---

## Text Generation

After training, the model generates text autoregressively and produces more coherent output compared to the bigram model.

---

## Key Learnings

- Limitations of n-gram models  
- Working of self-attention mechanisms  
- Importance of positional embeddings  
- Role of Transformer blocks in language modeling  

---

## References

- Andrej Karpathy — nanoGPT  
- Vaswani et al., *Attention Is All You Need*  

---

## Requirements

- Python 3.x  
- PyTorch  

---

## How to Run

```bash
python bigram.py
python gpt.py



