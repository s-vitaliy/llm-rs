# Minimal LLM Inference Engine in Rust (Educational MVP)

![Static Badge](https://img.shields.io/badge/vibecoded-8A2BE2)

**Status:** Work in Progress  
**Goal:** Build a minimal, readable, CPU-only LLM inference engine in Rust for deep understanding of decoder-only transformer inference.

---

## 🎯 Project Goal

The purpose of this project is to gain hands-on, working knowledge of large language model (LLM) inference by building everything from scratch.  
**This is an educational project and not intended for production or high-performance use.**

You will:
- Learn how decoder-only inference works (attention, KV-cache, autoregressive generation, stateful loops)
- Explore where Rust helps or hinders low-level numerical implementation
- Focus on correctness and clarity above all else

---

## 🧑‍💻 Development Process

This project follows a strict, milestone-driven [7-day plan](development.md) (≈30 hours), with each day targeting a clear functional block:

| Day | Focus                        |
|-----|------------------------------|
| 1-2 | Math primitives, RoPE        |
| 3-4 | Attention, KV-cache, FFN     |
| 5   | Model and session pipeline   |
| 6   | Weight loading, validation   |
| 7   | Tokenizer, e2e demo, docs    |

**See [`development.md`](development.md) for detailed daily tasks, acceptance criteria, and solutions for common challenges.**

---

## 🛠️ How to Build & Run

### 1. Build the project

```shell
cargo build --release
```

### 2. Run all tests (highly recommended)

```shell
cargo test --all
```

### 3. Export PyTorch weights
Use the provided Python script (see [`scripts/export_weights.py`](scripts/export_weights.py)) to generate weights for a *tiny* 2-layer transformer matching the config below.

### 4. Run the example

```shell
cargo run --example basic
```

---

## 🗂️ Model Specification (Fixed for MVP)

- Decoder-only transformer (LLaMA style)
- 2 layers, 4 heads, model_dim=256, head_dim=64, ffn_dim=1024
- Vocab size: ~16k (hardcoded/stub for MVP)
- **CPU only**, FP32, single-threaded by default

---

## 🧪 Testing & Validation

- **Unit tests** on all math primitives
- Shape and invariant checks in KV-cache
- "Golden" integration test: output logits must match PyTorch reference (to 1e-4 tolerance)
- No “looks reasonable”/qualitative testing

---

## 📏 Unsafe Code Policy

- `unsafe` **only** allowed in small, isolated numerical kernels
- Always commented for invariants and safety reasoning
- Never in high-level logic

---

## 📚 Philosophy & Agent Guidelines

- Favor **local reasoning** over abstraction
- All code is easy to follow: _every operation can be explained_
- Minimal diffs, test coverage before refactor
- If in doubt, ask/simplify/clarify
- _Never_ generalize or optimize prematurely

---

## 🚩 Red Flags to Avoid

- "Let's generalize this" / "this will be useful later"
- "For performance reasons"
- "Production-ready"
- "Best practice" or "modularize..."

> **This is a numerical lab, not a framework.**

---

## 📄 License

GPL-3.0 License. See [LICENSE](LICENSE).

---

**If a change makes the code harder to explain, it's probably wrong for this project.**

---
