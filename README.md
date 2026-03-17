# miniGPT — Character-Level Language Model in PyTorch

Built a decoder-only transformer from scratch in PyTorch, trained on the 
Tiny Shakespeare dataset. Started with a simple bigram baseline, then 
incrementally added self-attention, multi-head attention, positional 
embeddings, and transformer blocks to arrive at a GPT-style architecture.

## Results

**Bigram model** (loss: 4.73 → 2.46, single character context):
```
CEThik brid owindakis b, bth
HAPet bobe d e.
S:
O:3 my d?
```
No real words, no structure. Model has zero memory beyond one character.

**GPT model** (loss: 4.22 → 1.49, 256 token context, 10.8M parameters):
```
ANGELO:
Why, come, go then; and my lords;
For I shall saughter; for thou here were so strept,
Thou dost all men to the babstardness.

PAGE:
Our honest hange, mighty use,
choosed more again.
```
Learned play structure, character names, and dialogue format purely 
from character-level prediction.

## Architecture
```
Input tokens
     ↓
Token Embeddings + Positional Embeddings
     ↓
6x Transformer Blocks:
   - Multi-Head Self-Attention (6 heads, causal masking)
   - Feedforward Network (4x hidden dim, ReLU)
   - Residual connections + Layer Norm
     ↓
Linear projection → vocab logits → softmax → next token
```

~10.8M parameters total.

## What I learned

- Why bigram models fail: no memory beyond one character
- How Q, K, V work: query asks "what am I looking for",
  key says "what do I contain", value is what gets passed forward
- Why positional embeddings are needed: attention has no
  inherent sense of order
- How residual connections + layer norm stabilize deep networks
- Why dropout prevents overfitting in deep architectures

## Files

| File | Description |
|------|-------------|
| `bigram.py` | Baseline bigram model |
| `gpt.py` | Full GPT implementation |
| `input.txt` | Tiny Shakespeare dataset |
| `results/` | Sample generated outputs |

## Hyperparameters (GPT)

| Parameter | Value |
|-----------|-------|
| Context length | 256 tokens |
| Batch size | 64 |
| Embedding dim | 384 |
| Attention heads | 6 |
| Transformer layers | 6 |
| Dropout | 0.2 |
| Optimizer | AdamW (lr=3e-4) |
| Training steps | 2000 |

## How to run
```bash
# Clone the repo
git clone https://github.com/Kassu-GIT/miniGPT
cd miniGPT

# Install dependencies
pip install torch

# Run bigram baseline
python bigram.py

# Run full GPT
python gpt.py
```

## References
- Andrej Karpathy — [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)



