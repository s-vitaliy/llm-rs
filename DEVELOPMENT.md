# 7-Day Development Plan

**Project:** Minimal LLM Inference Engine in Rust  
**Timeline:** 7 days (~30 hours total)  
**Target:** Educational MVP with CPU-only inference

---

## 📊 Overview

| Milestone | Days | Focus Area |
|-----------|------|------------|
| **M1: Math Foundation** | 1-2 | Numerical primitives & RoPE |
| **M2: Core Attention** | 3-4 | KV-cache, Attention, FFN |
| **M3: Full Pipeline** | 5 | Multi-layer model & generation |
| **M4: Integration** | 6-7 | PyTorch weights & tokenizer |

---

## Day 1: Foundation & Math Primitives

**Goal:** Set up project structure and implement basic numerical operations.

**Time Budget:** 4 hours (2.5h coding, 1h testing, 0.5h docs)

### Tasks

#### 1.1 Project Structure
- [ ] Create `src/math/` directory
- [ ] Create `src/model/` directory
- [ ] Create `src/runtime/` directory
- [ ] Create `src/io/` directory
- [ ] Create `src/tokenizer/` directory
- [ ] Update `src/lib.rs` with module declarations

#### 1.2 Type-Safe Matrix Operations (`src/math/matrix.rs`)
- [ ] Define `struct Matrix<const ROWS: usize, const COLS: usize> { data: Vec<f32> }`
- [ ] Implement `Matrix::new(data: Vec<f32>) -> Result<Self, ShapeError>` with length validation
- [ ] Implement `Matrix::from_slice(data: &[f32]) -> Result<Self, ShapeError>`
- [ ] Implement `Mul<Matrix<K, N>>` for `Matrix<M, K>` returning `Matrix<M, N>`
  - [ ] Compiler guarantees dimension K matches
  - [ ] Invalid operations don't compile
- [ ] Implement `Matrix::matvec(&self, vec: &[f32; COLS]) -> [f32; ROWS]`
- [ ] Write unit test: `Matrix<2, 3> * Matrix<3, 2> -> Matrix<2, 2>`
- [ ] Write unit test: invalid multiplication doesn't compile (doc-test)
- [ ] Verify results against hand-calculated values

#### 1.3 Softmax (`src/math/softmax.rs`)
- [ ] Implement `fn softmax(logits: &[f32]) -> Vec<f32>`
- [ ] Use numerically stable version (subtract max before exp)
- [ ] Write unit test with simple input: `[1.0, 2.0, 3.0]`
- [ ] Write unit test with large values to verify stability
- [ ] Verify output sums to 1.0 (within epsilon)

#### 1.4 RMSNorm (`src/math/rmsnorm.rs`)
- [ ] Implement `fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32>`
- [ ] Calculate RMS: `sqrt(mean(x^2) + eps)`
- [ ] Normalize: `x / rms`
- [ ] Scale by weight: `normalized * weight`
- [ ] Write unit test with simple vectors
- [ ] Compare against PyTorch calculation (manual)

### Acceptance Criteria
- [ ] All math operations have unit tests
- [ ] Tests pass with epsilon tolerance (1e-5)
- [ ] Code compiles without warnings
- [ ] Functions have doc comments with usage examples

### Notes / Blockers
```
(track issues here)
```

---

## Day 2: Attention Mechanism (Part 1)

**Goal:** Implement RoPE and prepare attention scaffolding.

**Time Budget:** 3.5 hours (2h coding, 1h testing, 0.5h docs)

### Tasks

#### 2.1 Rotary Position Embeddings (`src/math/rope.rs`)
- [ ] Implement `fn compute_freqs(dim: usize, max_seq_len: usize, theta: f32) -> Vec<f32>`
- [ ] Implement `fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, freqs: &[f32])`
- [ ] Rotate pairs of elements: `(x, y) -> (x*cos - y*sin, x*sin + y*cos)`
- [ ] Write unit test with pos=0 (should be identity)
- [ ] Write unit test with known angle (e.g., 90°)
- [ ] Verify Q and K are rotated correctly

#### 2.2 Model Configuration (`src/model/config.rs`)
- [ ] Define generic `struct ModelConfig<const N_LAYERS, const MODEL_DIM, const N_HEADS, const FFN_DIM, const VOCAB_SIZE>`:
  - [ ] `rope_theta: f32` (= 10000.0)
  - [ ] `max_seq_len: usize` (= 512)
  - [ ] `eps: f32` (= 1e-5)
  - [ ] `HEAD_DIM` computed as `MODEL_DIM / N_HEADS` (compile-time check)
- [ ] Define type alias for MVP configuration:
  - [ ] `type TinyLlamaConfig = ModelConfig<2, 256, 4, 1024, 16384>`
- [ ] Implement `const fn validate()` for compile-time checks:
  - [ ] `MODEL_DIM % N_HEADS == 0`
  - [ ] `N_LAYERS > 0`
- [ ] Add helper const functions: `head_dim()`, `kv_dim()`

#### 2.3 Weight Structures (`src/model/weights.rs`)
- [ ] Define generic `struct AttentionWeights<const D: usize>`:
  - [ ] `wq: Matrix<D, D>`
  - [ ] `wk: Matrix<D, D>`
  - [ ] `wv: Matrix<D, D>`
  - [ ] `wo: Matrix<D, D>`
- [ ] Define generic `struct FfnWeights<const D: usize, const F: usize>`:
  - [ ] `w1: Matrix<D, F>` // gate
  - [ ] `w2: Matrix<D, F>` // up
  - [ ] `w3: Matrix<F, D>` // down
- [ ] Define generic `struct LayerWeights<const D: usize, const F: usize>`:
  - [ ] `attention: AttentionWeights<D>`
  - [ ] `ffn: FfnWeights<D, F>`
  - [ ] `attention_norm: [f32; D]`
  - [ ] `ffn_norm: [f32; D]`
- [ ] Types guarantee correct dimensions at compile time

### Acceptance Criteria
- [ ] RoPE tests pass for known rotation angles
- [ ] Config validates dimensions (e.g., model_dim = n_heads * head_dim)
- [ ] Weight structures compile and can be instantiated
- [ ] All code documented

### Notes / Blockers
```
(track issues here)
```

---

## Day 3: KV-Cache & Attention (Part 2)

**Goal:** Implement stateful KV-cache and complete attention mechanism.

**Time Budget:** 5 hours (3h coding, 1.5h testing, 0.5h docs)

### Tasks

#### 3.1 KV-Cache (`src/runtime/kv_cache.rs`)
- [ ] Define `struct KVCache`:
  - [ ] `keys: Vec<Vec<Vec<f32>>>` // [n_layers][seq_len][n_heads * head_dim]
  - [ ] `values: Vec<Vec<Vec<f32>>>`
  - [ ] `config: ModelConfig`
- [ ] Implement `fn new(config: ModelConfig) -> Self`
- [ ] Implement `fn append(&mut self, layer: usize, k: Vec<f32>, v: Vec<f32>)`
- [ ] Implement `fn get_keys(&self, layer: usize) -> &[Vec<f32>]`
- [ ] Implement `fn get_values(&self, layer: usize) -> &[Vec<f32>]`
- [ ] Implement `fn current_seq_len(&self, layer: usize) -> usize`
- [ ] Add shape validation in `append`
- [ ] Write unit test: append and retrieve for single layer
- [ ] Write unit test: verify cache grows correctly

#### 3.2 Attention Implementation (`src/model/attention.rs`)
- [ ] Create file with attention logic
- [ ] Implement `fn attention(x: &[f32], weights: &AttentionWeights, kv_cache: &mut KVCache, layer: usize, pos: usize, config: &ModelConfig) -> Vec<f32>`
- [ ] Step 1: Compute Q, K, V projections (use matmul)
  - [ ] `q = wq @ x` // [model_dim]
  - [ ] `k = wk @ x`
  - [ ] `v = wv @ x`
- [ ] Step 2: Apply RoPE to Q and K
  - [ ] Call `apply_rope(&mut q, &mut k, pos, ...)`
- [ ] Step 3: Append K, V to cache
  - [ ] `kv_cache.append(layer, k, v)`
- [ ] Step 4: Compute attention scores
  - [ ] Reshape Q to [n_heads, head_dim]
  - [ ] For each head:
    - [ ] Get all keys for this head from cache
    - [ ] Compute scores: `q_head @ k_head^T / sqrt(head_dim)`
    - [ ] Apply softmax over sequence dimension
    - [ ] Compute weighted sum: `scores @ v_head`
  - [ ] Concatenate all heads
- [ ] Step 5: Output projection
  - [ ] `output = wo @ concatenated`
- [ ] Write unit test with dummy weights (identity matrices)
- [ ] Write unit test: verify KV-cache contains correct values after attention

### Acceptance Criteria
- [ ] KV-cache tests pass
- [ ] Attention forward pass completes without panic
- [ ] Cache grows by 1 token after each attention call
- [ ] Output shape is [model_dim]
- [ ] Can verify attention pattern on tiny example

### Notes / Blockers
```
(track issues here)
```

---

## Day 4: Feed-Forward Network & Layer

**Goal:** Complete transformer layer (attention + FFN).

**Time Budget:** 4 hours (2.5h coding, 1h testing, 0.5h docs)

### Tasks

#### 4.1 Feed-Forward Network (`src/model/ffn.rs`)
- [ ] Create file for FFN
- [ ] Implement `fn silu(x: f32) -> f32` (SiLU activation)
  - [ ] Formula: `x / (1 + exp(-x))`
- [ ] Implement `fn ffn(x: &[f32], weights: &FfnWeights, config: &ModelConfig) -> Vec<f32>`
- [ ] Step 1: Gate projection
  - [ ] `gate = w1 @ x` // [ffn_dim]
  - [ ] Apply SiLU element-wise
- [ ] Step 2: Up projection
  - [ ] `up = w2 @ x` // [ffn_dim]
- [ ] Step 3: Element-wise multiply
  - [ ] `hidden = silu(gate) ⊙ up`
- [ ] Step 4: Down projection
  - [ ] `output = w3 @ hidden` // [model_dim]
- [ ] Write unit test with random weights
- [ ] Verify output shape

#### 4.2 Transformer Layer (`src/model/layer.rs`)
- [ ] Create file for full layer
- [ ] Implement `fn transformer_layer(x: &[f32], weights: &LayerWeights, kv_cache: &mut KVCache, layer: usize, pos: usize, config: &ModelConfig) -> Vec<f32>`
- [ ] Step 1: Pre-attention RMSNorm
  - [ ] `normed = rmsnorm(x, weights.attention_norm, config.eps)`
- [ ] Step 2: Self-attention with residual
  - [ ] `attn_out = attention(normed, &weights.attention, kv_cache, layer, pos, config)`
  - [ ] `x = x + attn_out`
- [ ] Step 3: Pre-FFN RMSNorm
  - [ ] `normed = rmsnorm(x, weights.ffn_norm, config.eps)`
- [ ] Step 4: FFN with residual
  - [ ] `ffn_out = ffn(normed, &weights.ffn, config)`
  - [ ] `x = x + ffn_out`
- [ ] Return final output
- [ ] Write integration test: full layer with random weights

### Acceptance Criteria
- [ ] FFN tests pass
- [ ] Full transformer layer runs without panic
- [ ] Residual connections preserve shape
- [ ] Can run 2 sequential layers and verify cache growth

### Notes / Blockers
```
(track issues here)
```

---

## Day 5: Full Model & Session Loop

**Goal:** Implement multi-layer forward pass and autoregressive generation.

**Time Budget:** 4.5 hours (3h coding, 1h testing, 0.5h docs)

### Tasks

#### 5.1 Model Structure (`src/model/model.rs`)
- [ ] Define `struct Model`:
  - [ ] `config: ModelConfig`
  - [ ] `embeddings: Vec<f32>` // [vocab_size, model_dim]
  - [ ] `layers: Vec<LayerWeights>`
  - [ ] `output_norm: Vec<f32>` // [model_dim]
  - [ ] `output_proj: Vec<f32>` // [model_dim, vocab_size]
- [ ] Implement `fn new_random(config: ModelConfig) -> Self`
  - [ ] Initialize all weights with random values (for testing)
- [ ] Implement `fn forward_step(&self, token_id: usize, kv_cache: &mut KVCache, pos: usize) -> Vec<f32>`
- [ ] Step 1: Embedding lookup
  - [ ] `x = embeddings[token_id * model_dim .. (token_id+1) * model_dim]`
- [ ] Step 2: Loop through all layers
  - [ ] `for (layer_idx, layer_weights) in layers.iter().enumerate()`
  - [ ] `x = transformer_layer(x, layer_weights, kv_cache, layer_idx, pos, &config)`
- [ ] Step 3: Final RMSNorm
  - [ ] `x = rmsnorm(x, &output_norm, config.eps)`
- [ ] Step 4: Output projection to vocabulary
  - [ ] `logits = output_proj @ x` // [vocab_size]
- [ ] Return logits
- [ ] Write unit test: forward step with dummy model

#### 5.2 Session & Generation (`src/runtime/session.rs`)
- [ ] Define `struct Session`:
  - [ ] `model: Model`
  - [ ] `kv_cache: KVCache`
- [ ] Implement `fn new(model: Model) -> Self`
- [ ] Implement `fn generate(&mut self, prompt_tokens: &[usize], max_new_tokens: usize) -> Vec<usize>`
- [ ] Step 1: Process prompt tokens
  - [ ] `for (pos, &token) in prompt_tokens.iter().enumerate()`
  - [ ] `logits = self.model.forward_step(token, &mut self.kv_cache, pos)`
- [ ] Step 2: Autoregressive generation
  - [ ] `let mut generated = Vec::new()`
  - [ ] `let mut pos = prompt_tokens.len()`
  - [ ] `for _ in 0..max_new_tokens`
    - [ ] Get last logits
    - [ ] `next_token = argmax(logits)`
    - [ ] `generated.push(next_token)`
    - [ ] `logits = self.model.forward_step(next_token, &mut self.kv_cache, pos)`
    - [ ] `pos += 1`
- [ ] Return generated tokens
- [ ] Implement `fn argmax(logits: &[f32]) -> usize` helper
- [ ] Write integration test: generate 5 tokens from prompt

### Acceptance Criteria
- [ ] Full model forward pass completes
- [ ] Generation loop produces tokens
- [ ] KV-cache size equals prompt_len + generated_len
- [ ] Output is deterministic (same input -> same output)

### Notes / Blockers
```
(track issues here)
```

---

## Day 6: Weight Loading & PyTorch Validation

**Goal:** Load real weights from PyTorch and validate numerical correctness.

**Time Budget:** 4.5 hours (2h coding, 2h debugging, 0.5h docs)

### Tasks

#### 6.1 PyTorch Export Script (`scripts/export_weights.py`)
- [ ] Create Python script with PyTorch
- [ ] Define tiny 2-layer transformer with exact config:
  - [ ] n_layers=2, n_heads=4, model_dim=256, head_dim=64, ffn_dim=1024
- [ ] Initialize with fixed seed for reproducibility
- [ ] Export each weight as separate `.npy` file:
  - [ ] `embeddings.npy`
  - [ ] `layer_0_attn_wq.npy`, `layer_0_attn_wk.npy`, etc.
  - [ ] `layer_0_ffn_w1.npy`, etc.
  - [ ] `output_norm.npy`
  - [ ] `output_proj.npy`
- [ ] Save test input tokens: `test_input.npy`
- [ ] Run forward pass and save logits: `test_logits.npy`
- [ ] Run script and verify all files generated

#### 6.2 Type-Safe Weight Loader (`src/io/npy_loader.rs`)
- [ ] Implement `.npy` file format parser
- [ ] Parse header to get dtype and shape
- [ ] Implement `fn load_npy(path: &str) -> Result<Vec<f32>, LoadError>`
- [ ] **Parse, don't validate:** Implement loading into type-safe model
  - [ ] `fn load_matrix<const R: usize, const C: usize>(path: &str) -> Result<Matrix<R, C>, LoadError>`
  - [ ] Validation: file size == R * C * sizeof(f32)
  - [ ] If dimensions don't match - return load error, not runtime panic
  - [ ] After loading, impossible to create matrix with wrong size
- [ ] Implement `fn load_model<C: ModelConfig>(dir: &str) -> Result<Model<C>, LoadError>`
  - [ ] Use const generics from `C` to determine sizes
  - [ ] Load each matrix with type-safe dimensions
  - [ ] Validation happens once during loading
  - [ ] Returned `Model<C>` guarantees correctness at type level
- [ ] Define error type `LoadError` with detailed messages:
  - [ ] `InvalidShape { expected: (usize, usize), got: (usize, usize), file: String }`
  - [ ] `IoError`, `ParseError`
- [ ] Write unit test: load simple npy file into `Matrix<3, 3>`

#### 6.3 Golden Test (`tests/golden_test.rs`)
- [ ] Create integration test file
- [ ] Load model from exported weights
- [ ] Load test input tokens
- [ ] Load expected logits from PyTorch
- [ ] Run forward pass in Rust
- [ ] Compare logits element-wise with epsilon=1e-4
- [ ] If mismatch:
  - [ ] Print first 10 values of each
  - [ ] Check for transpose issues
  - [ ] Verify weight loading order
  - [ ] Add debug prints in attention
  - [ ] Compare intermediate values (Q, K, V)

### Acceptance Criteria
- [ ] PyTorch script successfully exports weights
- [ ] Rust loads all weights without errors
- [ ] Logits match PyTorch within 1e-4 tolerance
- [ ] Test is deterministic and reproducible

### Notes / Blockers
```
(Debug notes for numerical mismatches)
```

---

## Day 7: Tokenizer & End-to-End Demo

**Goal:** Add minimal tokenizer and complete working demo.

**Time Budget:** 4 hours (2h coding, 1h testing, 1h docs)

### Tasks

#### 7.1 Simple Tokenizer (`src/tokenizer/simple.rs`)
- [ ] Choose approach: character-level or simple BPE
- [ ] Implement `struct Tokenizer`:
  - [ ] `vocab: HashMap<String, usize>`
  - [ ] `reverse_vocab: HashMap<usize, String>`
- [ ] Implement `fn new() -> Self` (hardcoded vocab for now)
- [ ] Implement `fn encode(&self, text: &str) -> Vec<usize>`
  - [ ] Greedy longest-match tokenization
  - [ ] Handle unknown tokens (map to UNK)
- [ ] Implement `fn decode(&self, tokens: &[usize]) -> String`
  - [ ] Concatenate token strings
- [ ] Write unit test: "hello" -> [token_ids] -> "hello"

#### 7.2 End-to-End Example (`examples/basic.rs`)
- [ ] Create example file
- [ ] Load model from weights directory
  - [ ] `let model = load_model_weights("./weights", &config)?;`
- [ ] Initialize session
  - [ ] `let mut session = Session::new(model);`
- [ ] Create tokenizer
  - [ ] `let tokenizer = Tokenizer::new();`
- [ ] Encode prompt
  - [ ] `let prompt = "Hello world";`
  - [ ] `let tokens = tokenizer.encode(prompt);`
- [ ] Generate tokens
  - [ ] `let generated = session.generate(&tokens, 20);`
- [ ] Decode output
  - [ ] `let output = tokenizer.decode(&generated);`
- [ ] Print results
  - [ ] Print prompt, generated tokens (as integers), decoded text
- [ ] Run with `cargo run --example basic`

#### 7.3 Polish & Documentation
- [ ] Clean up compiler warnings
- [ ] Add doc comments to all public functions
- [ ] Update `README.md` with:
  - [ ] Project description
  - [ ] Build instructions
  - [ ] How to export weights from PyTorch
  - [ ] How to run example
- [ ] Run `cargo test --all` and verify all tests pass
- [ ] Run `cargo clippy` and fix suggestions
- [ ] Format code with `rustfmt`

### Acceptance Criteria
- [ ] Example runs successfully from command line
- [ ] Generated text is deterministic
- [ ] All tests pass (`cargo test`)
- [ ] No compiler warnings
- [ ] README is clear and complete

### Notes / Blockers
```
(final notes)
```

---

## 📈 Progress Tracking

### Milestone 1: Math Foundation (Days 1-2)
- [ ] All math primitives implemented
- [ ] All unit tests passing
- [ ] RoPE working correctly
- [ ] Model config and weight structures defined

### Milestone 2: Core Attention (Days 3-4)
- [ ] KV-cache implementation complete
- [ ] Attention mechanism working
- [ ] FFN implementation complete
- [ ] Full transformer layer working

### Milestone 3: Full Pipeline (Day 5)
- [ ] Multi-layer model implemented
- [ ] Autoregressive generation loop working
- [ ] Can generate tokens deterministically

### Milestone 4: Integration (Days 6-7)
- [ ] PyTorch weights successfully loaded
- [ ] Logits match PyTorch reference
- [ ] Tokenizer implemented
- [ ] End-to-end example working

---

## 🎯 Definition of Done

The MVP is complete when ALL of the following are true:

- [ ] `cargo build --release` completes without errors or warnings
- [ ] `cargo test --all` passes all tests
- [ ] `cargo run --example basic` generates text from a prompt
- [ ] Golden test passes (logits match PyTorch within 1e-4)
- [ ] KV-cache correctly maintains state across tokens
- [ ] Same input always produces same output (deterministic)
- [ ] README.md has clear build and usage instructions
- [ ] Core functions have documentation comments

---

## 🚧 Known Challenges & Solutions

### Challenge: Matrix Dimension Mismatches
**Symptoms:** Panics in matmul, wrong output shapes  
**Solution:**
- [ ] Add explicit shape assertions at function entry
- [ ] Create `debug_shape()` helper that prints dimensions
- [ ] Draw out dimensions on paper before implementing

### Challenge: KV-Cache Indexing Errors
**Symptoms:** Index out of bounds, wrong attention output  
**Solution:**
- [ ] Test with very short sequences (3-5 tokens) first
- [ ] Print cache size after each append
- [ ] Draw cache structure diagram
- [ ] Add invariant checks in KVCache methods

### Challenge: PyTorch Weight Loading
**Symptoms:** Numerical mismatches in golden test  
**Solution:**
- [ ] Export PyTorch weights with verbose names
- [ ] Print first 5 values of each weight when loading
- [ ] Check if matrices need transposing
- [ ] Compare intermediate values (Q, K, V) not just final logits

### Challenge: Numerical Instability
**Symptoms:** NaN or Inf in attention or softmax  
**Solution:**
- [ ] Use stable softmax (subtract max)
- [ ] Add epsilon to denominators (1e-8)
- [ ] Check for NaN after each major operation
- [ ] Use `f32::is_finite()` in debug builds

---

## 📊 Time Tracking

| Day | Planned | Actual | Delta | Notes |
|-----|---------|--------|-------|-------|
| 1   | 4.0h    |        |       |       |
| 2   | 3.5h    |        |       |       |
| 3   | 5.0h    |        |       |       |
| 4   | 4.0h    |        |       |       |
| 5   | 4.5h    |        |       |       |
| 6   | 4.5h    |        |       |       |
| 7   | 4.0h    |        |       |       |
| **Total** | **29.5h** | | | |

---

## 🎓 Learning Objectives

By completing this plan, you will understand:

✅ How matrix multiplications compose to form attention  
✅ Why KV-cache is essential for efficient generation  
✅ How RoPE embeddings work at the implementation level  
✅ Where numerical instability occurs (softmax, attention scores)  
✅ How autoregressive generation maintains state  
✅ The difference between understanding transformers conceptually and implementing them  
✅ Where Rust's ownership model helps and where it creates friction  

---

## 📝 Daily Reflection Template

```markdown
### Day X - [Date]

**Time spent:** X hours

**Completed:**
- Task 1
- Task 2

**Challenges:**
- Issue 1 and how I resolved it
- Issue 2 (still blocked)

**Learnings:**
- Key insight 1
- Key insight 2

**Tomorrow:**
- Priority 1
- Priority 2
```

---

## 🔄 Adaptation Strategy

### If Ahead of Schedule
- [ ] Add temperature sampling (stochastic generation)
- [ ] Implement top-k or top-p sampling
- [ ] Add simple REPL for interactive chat
- [ ] Profile code and identify bottlenecks (don't optimize yet)
- [ ] Write detailed blog post explaining implementation

### If Behind Schedule
Priority 1 (keep):
- Math primitives (Days 1-2)
- Attention + KV-cache (Day 3)
- Full model (Day 5)

Priority 2 (simplify):
- Use synthetic random weights instead of PyTorch export
- Skip tokenizer, use integer arrays directly
- Reduce test coverage (but keep attention tests)

Priority 3 (drop if necessary):
- Golden test comparison with PyTorch
- Polished example
- Comprehensive documentation

---

**Start Date:** _____________  
**Target Completion:** _____________  
**Actual Completion:** _____________
