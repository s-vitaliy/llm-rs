# agents.md

## Project: Minimal LLM Inference Engine in Rust (Educational MVP)

### 🎯 Project Goal

Build a **minimal, readable, CPU-only LLM inference engine in Rust** in order to:

- deeply understand how decoder-only LLM inference works
- understand KV-cache, autoregressive generation, attention, and stateful inference
- explore where Rust helps and where it hurts in low-level numerical code

This project is **educational**, not a production system.

---

## 🚫 Explicit Non-Goals (Very Important)

The agent MUST NOT:

- introduce ggml, ggml-rs, llama.cpp bindings, or any external inference engine
- introduce GGUF loading in the MVP phase
- design a general-purpose tensor framework
- introduce SIMD, GPU, Metal, CUDA, Vulkan, or wgpu
- optimize for performance or tokens/sec
- refactor toward extensibility or “future-proofing”
- add abstractions “just in case”
- compare performance against llama.cpp
- attempt to be idiomatic at the cost of clarity

**Correctness, clarity, and understanding > performance and abstraction.**

---

## 🧱 Architectural Constraints (Hard Rules)

- CPU-only
- FP32 initially (FP16 may come later, but not now)
- Decoder-only transformer (LLaMA-style)
- Hardcoded inference pipeline (no computation graph)
- No generic `Tensor` abstraction in MVP
- Explicit loops over layers and operations
- Explicit KV-cache structure
- Deterministic execution (single-threaded unless explicitly stated)

---

## 📦 Model Scope (Fixed for MVP)

The MVP model is a **Tiny LLaMA-style transformer**:

- `n_layers = 2`
- `n_heads = 4`
- `model_dim = 256`
- `head_dim = 64`
- `ffn_dim = 1024`
- `vocab_size = 16k–32k`

The agent MUST NOT suggest larger models or architectural variants.

---

## 🗂️ Code Organization (Current Structure)

The project uses a flat module structure without `mod.rs` files:

```
src/
├── lib.rs              // Library entry point with module declarations
├── math.rs             // Math module declaration
├── math/
│   ├── matrix.rs       // Matrix type with compile-time shape checking
│   └── matrix/
│       └── tests.rs    // Matrix unit tests
├── model.rs            // Model architecture (stub)
├── runtime.rs          // Runtime state, KV-cache, session (stub)
├── io.rs               // I/O operations for loading weights (stub)
└── tokenizer.rs        // Tokenization utilities (stub)
```

### Module Organization Principles

- **No `mod.rs` files**: Use `module_name.rs` instead of `module_name/mod.rs`
- **Tests in submodules**: Tests are in `module_name/tests.rs` (e.g., `math/matrix/tests.rs`)
- **Flat structure**: Top-level modules are single `.rs` files that declare submodules

### Expected Evolution

As the project grows, the structure will expand to:

```
math/
  matrix.rs + matrix/tests.rs   // ✅ Implemented
  matmul.rs + matmul/tests.rs   // Planned: raw matrix multiplication kernels
  softmax.rs + softmax/tests.rs // Planned: softmax operation
  rmsnorm.rs + rmsnorm/tests.rs // Planned: RMS normalization
  rope.rs + rope/tests.rs        // Planned: rotary position embeddings

model/
  config.rs      // Planned: hyperparameters struct
  weights.rs     // Planned: strongly typed weight structures
  forward.rs     // Planned: forward pass implementation

runtime/
  kv_cache.rs    // Planned: KV-cache for attention
  state.rs       // Planned: inference state management
  session.rs     // Planned: autoregressive generation loop

io/
  npy_loader.rs  // Planned: load weights from NumPy/PyTorch exports

tokenizer/
  simple.rs      // Planned: minimal tokenizer implementation
```

The agent MUST NOT introduce:
- a graph executor
- dynamic dispatch for ops
- deep trait hierarchies

---

## 🧠 Development Philosophy

The agent should prioritize:

1. **Transparency**  
   Every operation should be easy to follow and explain.

2. **Local Reasoning**  
   Prefer explicit code over abstraction layers.

3. **Numerical Correctness**  
   Shapes, ordering, and math must match reference implementations.

4. **Testability**  
   Small, deterministic units that can be compared to PyTorch.

---

## 🧪 Testing Expectations

The agent should prefer:

- unit tests on math primitives (matmul, softmax, RMSNorm)
- shape and invariant checks (especially KV-cache)
- golden tests comparing logits to PyTorch
- epsilon-based floating point comparisons

The agent MUST NOT rely on:
- “the model output looks reasonable”
- long prompt-based testing
- qualitative evaluation of generated text

---

## 🧨 Unsafe Code Policy

- `unsafe` is allowed **only** in clearly isolated numerical kernels
- each `unsafe` block must have:
  - a comment explaining why it is safe
  - a comment explaining what invariant it relies on
- the agent MUST NOT spread `unsafe` across high-level logic

---

## 🧭 Agent Behavior Guidelines

When proposing code or changes, the agent should:

- explain *why* the change exists
- explicitly state which part of the inference pipeline it affects
- prefer minimal diffs over refactors
- ask before expanding scope

If uncertain, the agent should:
- ask a clarifying question
- or propose the simplest possible version first

---

## 🛑 Red Flags (Agent Should Avoid)

- “Let’s generalize this”
- “This will be useful later”
- “For performance reasons”
- “Production-ready”
- “Industry best practice”
- “Let’s make it modular”

These are usually **wrong for this project**.

---

## ✅ Definition of MVP Done

The MVP is complete when:

- a token-by-token autoregressive loop runs
- KV-cache grows correctly per token
- logits are produced deterministically
- generation works with real (exported) weights
- results can be compared to PyTorch for correctness

The model does NOT need to be:
- fast
- smart
- user-friendly

---

## 🧠 Mental Model to Keep

This project is a **numerical laboratory**, not a framework.

If a change makes the code harder to explain to another engineer,
it is probably the wrong change.
