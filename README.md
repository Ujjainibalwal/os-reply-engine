# 🚀 AI Reply Engine — Summer Project Roadmap (Revised)

## Training Your Own Language Model from Scratch + Building a Reply Engine

---

## ⚠️ Reality Check: Training from Scratch on a Laptop

> [!CAUTION]
> Training a full-sized LLM (like GPT-3, Llama 3) from scratch requires **thousands of GPUs running for weeks** and costs **millions of dollars**. This is NOT feasible for a summer project on any budget.

> [!IMPORTANT]
> **But here's the good news** — you CAN train a **small language model from scratch** that is academically meaningful and demonstrates you understand the entire pipeline. Here's what's realistic:

### What IS Feasible on a Laptop (No GPU)

| Model Type | Parameters | Training Time | Feasibility |
|-----------|-----------|--------------|-------------|
| **Tiny GPT (character-level)** | ~1M params | 1–2 hours on CPU | ✅ Very feasible |
| **Small GPT-2 style** | ~10–50M params | 6–24 hours on CPU | ✅ Feasible |
| **nanoGPT** | ~85M params | 2–5 days on CPU, few hours on Colab GPU | ✅ Feasible with Colab |
| GPT-2 (124M) | 124M params | Weeks on CPU | ⚠️ Use free Colab T4 |
| Anything >500M | 500M+ params | Impractical on CPU | ❌ Not feasible |

### 🎯 Recommended Approach for Your Project

**Train a small transformer model (10–85M parameters) from scratch**, then build a reply engine around it. Supplement it with a retrieval system (search/RAG) to make responses actually useful.

**Your project will demonstrate:**
1. ✅ You understand transformer architecture from the ground up
2. ✅ You can prepare datasets, tokenize text, and train models
3. ✅ You can build an end-to-end inference pipeline
4. ✅ You can deploy and serve a custom model
5. ✅ You understand the full ML lifecycle

---

## 📚 Phase 0: What You Must Study (Weeks 1–3)

> [!IMPORTANT]
> This is the most critical phase. You need solid foundations in deep learning, NLP, and transformers before you can train anything meaningful.

### Layer 1: Python & Math Foundations (Week 1)

| Topic | What to Study | Resource |
|-------|--------------|----------|
| **Python (advanced)** | OOP, generators, decorators, type hints | [Real Python](https://realpython.com/) |
| **NumPy** | Array operations, broadcasting, linear algebra | [NumPy tutorial](https://numpy.org/doc/stable/user/quickstart.html) |
| **Linear Algebra** | Matrix multiplication, dot products, eigenvalues | [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) (YouTube) |
| **Calculus basics** | Derivatives, chain rule, gradients | [3Blue1Brown — Essence of Calculus](https://www.3blue1brown.com/topics/calculus) (YouTube) |
| **Probability** | Distributions, Bayes' theorem, softmax | Khan Academy |

> [!TIP]
> You don't need to be a math expert. Focus on **intuition** — understand *what* matrix multiplication does and *why* gradients flow backward. The libraries handle the computation.

### Layer 2: Deep Learning & PyTorch (Week 2)

| Topic | What to Study | Resource |
|-------|--------------|----------|
| **Neural Networks** | Perceptrons, activation functions, backpropagation | [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| **PyTorch** | Tensors, autograd, nn.Module, DataLoader, training loops | [PyTorch official tutorials](https://pytorch.org/tutorials/) |
| **Loss Functions** | Cross-entropy loss (this is what language models use) | PyTorch docs |
| **Optimizers** | SGD, Adam, AdamW, learning rate scheduling | PyTorch docs |
| **GPU basics** | CUDA, `.to(device)`, mixed precision | PyTorch docs |

**Hands-on exercise:** Build and train a simple image classifier (MNIST) in PyTorch from scratch. This teaches you the training loop pattern you'll reuse for the language model.

### Layer 3: NLP & Transformers (Week 3)

| Topic | What to Study | Resource |
|-------|--------------|----------|
| **Tokenization** | BPE, WordPiece, SentencePiece | [HuggingFace Tokenizer docs](https://huggingface.co/docs/tokenizers/) |
| **Word Embeddings** | Word2Vec, token embeddings, positional encodings | [Jay Alammar's blog](https://jalammar.github.io/illustrated-word2vec/) |
| **Attention Mechanism** | Self-attention, multi-head attention, QKV | [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) ⭐ |
| **Transformer Architecture** | Encoder, decoder, layer norm, feed-forward | [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762) |
| **GPT Architecture** | Decoder-only transformer, causal masking, autoregressive generation | [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT) ⭐ |
| **The "Let's build GPT" video** | Build a GPT from scratch in code | [Andrej Karpathy's video](https://www.youtube.com/watch?v=kCc8FmEb1nY) ⭐⭐⭐ |

> [!IMPORTANT]
> **Andrej Karpathy's "Let's build GPT from scratch" video is THE single most important resource for your project.** Watch it, code along, and understand every line. It takes you from zero to a working GPT in ~2 hours.

---

## 🏗️ Phase 1: Build the Training Pipeline (Weeks 4–5)

### Step 1: Prepare Your Dataset

**What data to train on:**

| Dataset | Size | Best For | Source |
|---------|------|----------|--------|
| **TinyStories** | ~500MB | Small model that tells coherent stories | [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories) |
| **OpenWebText** | ~40GB (use a subset) | General web text, like GPT-2's training data | [HuggingFace](https://huggingface.co/datasets/openwebtext) |
| **Wikipedia (Simple English)** | ~200MB | Clean, factual text | [HuggingFace](https://huggingface.co/datasets/wikipedia) |
| **Custom domain data** | Varies | If your reply engine focuses on a specific topic | Scrape/collect yourself |

> [!TIP]
> Start with **TinyStories** — it's specifically designed for training small models and produces surprisingly coherent results even with <50M parameters.

**Data preparation pipeline:**
```
Raw Text → Clean & Filter → Tokenize (BPE) → Create Training Chunks → Save as Binary
```

```python
# Simplified data pipeline
from datasets import load_dataset
import tiktoken

# 1. Load dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")

# 2. Tokenize
enc = tiktoken.get_encoding("gpt2")  # Use GPT-2's tokenizer
tokens = []
for example in dataset:
    tokens.extend(enc.encode(example["text"]))

# 3. Save as numpy array for fast loading
import numpy as np
tokens = np.array(tokens, dtype=np.uint16)
tokens.tofile("train_data.bin")
```

### Step 2: Build the Model Architecture

**Your model (from scratch in PyTorch):**

```python
import torch
import torch.nn as nn
import math

class GPTConfig:
    block_size: int = 256      # context window (sequence length)
    vocab_size: int = 50257    # GPT-2 tokenizer vocab
    n_layer: int = 6           # number of transformer blocks
    n_head: int = 6            # number of attention heads
    n_embd: int = 384          # embedding dimension
    dropout: float = 0.2

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Causal mask
        self.register_buffer("mask",
            torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
```

**Model sizes you can experiment with:**

| Config Name | Layers | Heads | Embed Dim | ~Params | CPU Training Time |
|------------|--------|-------|-----------|---------|-------------------|
| **tiny** | 4 | 4 | 128 | ~1M | ~1 hour |
| **small** | 6 | 6 | 384 | ~10M | ~6–12 hours |
| **medium** | 8 | 8 | 512 | ~40M | ~2–4 days |
| **nanoGPT** | 12 | 12 | 768 | ~85M | Use Colab free tier |

### Step 3: Write the Training Loop

```python
# Training loop skeleton
config = GPTConfig()
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(max_steps):
    # Get batch
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log every 100 steps
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

    # Save checkpoint every 1000 steps
    if step % 1000 == 0:
        torch.save(model.state_dict(), f"checkpoint_{step}.pt")
```

**Key training concepts to implement:**
- [x] Gradient accumulation (simulate larger batch sizes on CPU)
- [x] Learning rate warmup + cosine decay schedule
- [x] Gradient clipping (prevent exploding gradients)
- [x] Validation loss tracking (detect overfitting)
- [x] Checkpoint saving (resume training after interruptions)
- [x] Mixed precision training (if using Colab GPU)

### Step 4: Text Generation (Inference)

```python
@torch.no_grad()
def generate(model, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop to block_size
        idx_cond = tokens[:, -config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k sampling
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return enc.decode(tokens[0].tolist())
```

---

## 🌐 Phase 2: Build the Reply Engine Around Your Model (Weeks 6–7)

Your custom model alone won't give great answers to factual questions (it doesn't have enough parameters/data). The trick is to combine it with a **retrieval system**.

### Architecture: Custom Model + Retrieval

```mermaid
graph LR
    A["👤 User Query"] --> B["🖥️ Frontend"]
    B --> C["⚙️ Backend API"]
    C --> D["🔍 Search/Retrieval Module"]
    D --> E["📄 Retrieved Context"]
    E --> F["📝 Prompt Builder"]
    F --> G["🧠 Your Custom Model"]
    G --> H["📝 Generated Response"]
    H --> B

    style G fill:#f5a623,stroke:#333,color:#000
```

### Backend API (FastAPI)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch

app = FastAPI(title="AI Reply Engine — Custom Model")

# Load your trained model
model = GPT(config)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

class Query(BaseModel):
    question: str

@app.post("/api/reply")
async def reply(query: Query):
    # 1. Search for relevant context (optional)
    context = await search_web(query.question)

    # 2. Build prompt with context
    prompt = f"""Context: {context}

Question: {query.question}
Answer:"""

    # 3. Generate with your model
    response = generate(model, prompt, max_new_tokens=300)

    return {"answer": response, "model": "custom-gpt-10M"}
```

### Frontend

Build the same clean UI from the original roadmap:
- Search bar for queries
- Streaming response display
- Markdown rendering
- Source citations (from the retrieval module)
- Dark mode, animations, polished design

**Tech**: Next.js or plain HTML/CSS/JavaScript

---

## 🚀 Phase 3: Deployment (Week 8–9)

### Serving Your Custom Model

Since your model is small (10–85M params), it can run on **CPU** — no GPU needed for inference!

| Deployment Option | Cost | Best For |
|------------------|------|----------|
| **Railway** (Docker) | Free tier | Easiest for small models |
| **Render** | Free tier | Similar to Railway |
| **HuggingFace Spaces** | Free | Great for demos |
| **VPS (DigitalOcean/Hetzner)** | $5–10/month | Full control |
| **Your college server** | Free | If available |

### Dockerize Everything

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model/ ./model/
COPY *.py .

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> [!TIP]
> Your model file will be small (~40–340MB for 10–85M params), so it fits easily in a Docker container and deploys to free tiers.

### Model Optimization for Deployment

```python
# Quantize model for faster CPU inference
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# This can make inference 2-4x faster on CPU
```

---

## 📅 Revised Week-by-Week Timeline (10 Weeks)

| Week | Focus | Deliverable |
|------|-------|-------------|
| **1** | Python, NumPy, linear algebra, calculus basics | Complete math/Python exercises |
| **2** | PyTorch fundamentals, build & train MNIST classifier | Working PyTorch training pipeline |
| **3** | NLP, tokenization, transformers, **watch Karpathy's video** | Understand transformer architecture fully |
| **4** | Prepare dataset, build tokenizer, write data pipeline | `train_data.bin` ready |
| **5** | Build model architecture, write training loop, **start training** | Model training on CPU/Colab |
| **6** | Monitor training, tune hyperparameters, implement generation | Working text generation from your model |
| **7** | Build backend API + search/retrieval module | `/api/reply` endpoint working |
| **8** | Build frontend UI | Full web app working locally |
| **9** | Deploy (Docker + Railway/Render), testing, bug fixes | Live deployed application |
| **10** | Documentation, project report, presentation, demo video | **Project submission ready** |

---

## 📚 Complete Study Resources

### 🌟 Must-Watch / Must-Read (in order)

| # | Resource | What You'll Learn | Time |
|---|----------|-------------------|------|
| 1 | [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Visual intuition for neural nets | 1 hour |
| 2 | [PyTorch in 60 Minutes](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | PyTorch fundamentals | 1 hour |
| 3 | [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | How transformers work (visual) | 30 min |
| 4 | ⭐ [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | **Build a GPT from scratch** — THE key video | 2 hours |
| 5 | [Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) | Production-quality small GPT training code | Study code |
| 6 | [Karpathy — Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | BPE tokenization from scratch | 2 hours |
| 7 | [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) | Backend API development | 2 hours |

### 📖 Books & Courses

| Resource | Topic | Type |
|----------|-------|------|
| [Deep Learning with PyTorch (free book)](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf) | PyTorch + deep learning | Book (free PDF) |
| [Stanford CS224N (YouTube)](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4) | NLP with Deep Learning | Lecture series |
| [fast.ai Part 2](https://course.fast.ai/Lessons/part2.html) | Build a language model from scratch | Course |
| [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course/) | Transformers, fine-tuning | Interactive |

### 🔧 Reference Repos to Study

| Repo | Why Study It |
|------|-------------|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Clean, minimal GPT training code (~300 lines) |
| [minGPT](https://github.com/karpathy/minGPT) | Even simpler GPT implementation |
| [llm.c](https://github.com/karpathy/llm.c) | GPT-2 training in pure C (advanced) |
| [TinyLlama](https://github.com/jzhang38/TinyLlama) | How a 1.1B model was trained on limited resources |

---

## 🧰 Tools & Setup

- [ ] **Python 3.10+** with virtual environment
- [ ] **PyTorch** (`pip install torch`)
- [ ] **tiktoken** (`pip install tiktoken`) — tokenizer
- [ ] **datasets** (`pip install datasets`) — HuggingFace datasets
- [ ] **wandb** (`pip install wandb`) — training metrics visualization (free)
- [ ] **Google Colab** — free T4 GPU for larger training runs
- [ ] **FastAPI + Uvicorn** — backend server
- [ ] **Docker** — containerization for deployment
- [ ] **Git + GitHub** — version control

---

## 💡 Tips for Your College Project

> [!TIP]
> ### How to Impress Your Faculty Guide
> 1. **Show the full pipeline**: data → tokenization → model → training → inference → deployment
> 2. **Include training curves**: Plot loss vs. steps (use wandb or matplotlib)
> 3. **Compare model sizes**: Train 1M, 10M, and 40M param models and compare outputs
> 4. **Show ablation studies**: What happens when you change layers, heads, learning rate?
> 5. **Demo live generation**: Let them type a prompt and see your model respond in real-time

> [!WARNING]
> ### Be Honest About Limitations
> Your small model will NOT match ChatGPT quality. That's expected and fine. In your report, explain:
> - Why larger models perform better (scaling laws)
> - What your model CAN do (generate coherent short text, complete patterns)
> - What it CANNOT do (factual accuracy, long reasoning, following complex instructions)
> - How the retrieval module compensates for model limitations

---

## Open Questions

1. **What domain should your model focus on?** — General text, stories, Q&A, technical docs, or something specific to your department?
2. **How long is your project timeline?** — Is 10 weeks realistic, or do you have more/less time?
3. **Does your faculty guide have specific requirements?** — E.g., must use a certain framework, must include a specific feature, must write a research paper format report?
4. **Google Colab** — Even though you don't have a GPU, would you use Google Colab's free T4 GPU for training? It would allow you to train a larger model (85M params) in a few hours instead of days.
