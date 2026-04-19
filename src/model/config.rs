//! Model configuration for the Tiny LLaMA-style transformer.

use crate::math::RopeTheta;

#[cfg(test)]
mod tests;

/// Validated maximum sequence length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxSeqLen(usize);

impl MaxSeqLen {
    /// Creates a validated maximum sequence length.
    ///
    /// # Panics
    ///
    /// Panics if `value` is zero.
    pub const fn new(value: usize) -> Self {
        assert!(value > 0, "max_seq_len must be greater than zero");
        Self(value)
    }

    /// Returns the underlying value.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Validated RMSNorm epsilon parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormEps(f32);

impl RmsNormEps {
    /// Creates a validated epsilon value.
    ///
    /// # Panics
    ///
    /// Panics if `value` is negative or non-finite.
    pub const fn new(value: f32) -> Self {
        assert!(
            value.is_finite() && value >= 0.0,
            "eps must be finite and >= 0.0"
        );
        Self(value)
    }

    /// Returns the underlying value.
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// Hyperparameters for a decoder-only transformer.
///
/// The architecture dimensions are encoded in const generics so shape
/// relationships can be validated close to the type itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig<
    const N_LAYERS: usize,
    const MODEL_DIM: usize,
    const N_HEADS: usize,
    const FFN_DIM: usize,
    const VOCAB_SIZE: usize,
> {
    rope_theta: RopeTheta,
    max_seq_len: MaxSeqLen,
    eps: RmsNormEps,
}

impl<
        const N_LAYERS: usize,
        const MODEL_DIM: usize,
        const N_HEADS: usize,
        const FFN_DIM: usize,
        const VOCAB_SIZE: usize,
    > ModelConfig<N_LAYERS, MODEL_DIM, N_HEADS, FFN_DIM, VOCAB_SIZE>
{
    const INVARIANTS: () = {
        assert!(N_LAYERS > 0, "N_LAYERS must be greater than zero");
        assert!(MODEL_DIM > 0, "MODEL_DIM must be greater than zero");
        assert!(N_HEADS > 0, "N_HEADS must be greater than zero");
        assert!(
            MODEL_DIM.is_multiple_of(N_HEADS),
            "MODEL_DIM must be divisible by N_HEADS"
        );
        assert!(FFN_DIM > 0, "FFN_DIM must be greater than zero");
        assert!(VOCAB_SIZE > 0, "VOCAB_SIZE must be greater than zero");
    };

    const HEAD_DIM: usize = {
        let () = Self::INVARIANTS;
        MODEL_DIM / N_HEADS
    };

    const KV_DIM: usize = {
        let () = Self::INVARIANTS;
        N_HEADS * Self::HEAD_DIM
    };

    /// Default runtime parameters for the MVP configuration style.
    pub const DEFAULT: Self = {
        let () = Self::INVARIANTS;
        Self {
            rope_theta: RopeTheta::new(10_000.0),
            max_seq_len: MaxSeqLen::new(512),
            eps: RmsNormEps::new(1e-5),
        }
    };

    /// Creates a config with explicit runtime parameters.
    ///
    /// The runtime parameters are already validated at construction time.
    pub fn new(rope_theta: RopeTheta, max_seq_len: MaxSeqLen, eps: RmsNormEps) -> Self {
        let () = Self::INVARIANTS;
        Self {
            rope_theta,
            max_seq_len,
            eps,
        }
    }

    /// Returns the per-head dimension.
    pub const fn head_dim() -> usize {
        Self::HEAD_DIM
    }

    /// Returns the concatenated key/value dimension.
    pub const fn kv_dim() -> usize {
        Self::KV_DIM
    }

    /// Returns the RoPE theta parameter.
    pub const fn rope_theta(&self) -> RopeTheta {
        self.rope_theta
    }

    /// Returns the configured maximum sequence length.
    pub const fn max_seq_len(&self) -> MaxSeqLen {
        self.max_seq_len
    }

    /// Returns the RMSNorm epsilon value.
    pub const fn eps(&self) -> RmsNormEps {
        self.eps
    }
}

/// Fixed MVP configuration from the development plan.
pub type TinyLlamaConfig = ModelConfig<2, 256, 4, 1024, 16_384>;
