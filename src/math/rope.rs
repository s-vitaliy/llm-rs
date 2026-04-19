//! Rotary Position Embeddings (RoPE).
//!
//! This module provides helpers for precomputing RoPE angles and applying the
//! rotation in place to query and key vectors.

#[cfg(test)]
mod tests;

/// Validated per-head dimension for RoPE.
///
/// RoPE operates on pairs of elements, so the head dimension must be positive
/// and even.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeHeadDim(usize);

impl RopeHeadDim {
    /// Creates a validated RoPE head dimension.
    ///
    /// # Panics
    ///
    /// Panics if `value` is zero or odd.
    pub fn new(value: usize) -> Self {
        assert!(value > 0, "head_dim must be greater than zero");
        assert!(value % 2 == 0, "head_dim must be even");
        Self(value)
    }

    /// Returns the underlying dimension value.
    pub fn get(self) -> usize {
        self.0
    }

    fn pair_count(self) -> usize {
        self.0 / 2
    }
}

/// Validated RoPE theta parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeTheta(f32);

impl RopeTheta {
    /// Creates a validated RoPE theta value.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not finite or is not positive.
    pub fn new(value: f32) -> Self {
        assert!(
            value.is_finite() && value > 0.0,
            "RoPE theta must be finite and positive"
        );
        Self(value)
    }

    fn get(self) -> f32 {
        self.0
    }
}

/// Precomputed RoPE angles for one head dimension across positions.
///
/// The underlying data is flattened in row-major order by position:
/// `[pos0_pair0, pos0_pair1, ..., pos1_pair0, pos1_pair1, ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeFreqs {
    head_dim: RopeHeadDim,
    data: Vec<f32>,
}

impl RopeFreqs {
    /// Creates validated RoPE frequencies from a flat angle buffer.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a whole number of positions for the given
    /// `head_dim`.
    pub fn from_angles(head_dim: RopeHeadDim, data: Vec<f32>) -> Self {
        assert!(
            data.len() % head_dim.pair_count() == 0,
            "RoPE frequency buffer must contain a whole number of positions"
        );

        Self { head_dim, data }
    }

    /// Returns the validated head dimension associated with these frequencies.
    pub fn head_dim(&self) -> RopeHeadDim {
        self.head_dim
    }

    /// Returns the number of stored positions.
    pub fn max_seq_len(&self) -> usize {
        self.data.len() / self.head_dim.pair_count()
    }

    /// Returns the number of stored angles.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the flat frequency buffer.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    fn angle(&self, pos: usize, pair_idx: usize) -> f32 {
        let pair_count = self.head_dim.pair_count();
        let freq_offset = pos
            .checked_mul(pair_count)
            .expect("position overflow while indexing RoPE frequencies");

        assert!(
            pair_idx < pair_count,
            "pair index out of bounds for head_dim"
        );
        assert!(
            freq_offset + pair_idx < self.data.len(),
            "RoPE frequencies do not contain the requested position"
        );

        self.data[freq_offset + pair_idx]
    }
}

/// Precomputes RoPE rotation angles for one attention head.
///
/// # Examples
///
/// ```
/// use llm_rs::math::{compute_freqs, RopeHeadDim, RopeTheta};
///
/// let freqs = compute_freqs(RopeHeadDim::new(4), 2, RopeTheta::new(10_000.0));
/// assert_eq!(freqs.len(), 4);
/// assert_eq!(freqs.as_slice()[0], 0.0);
/// assert_eq!(freqs.as_slice()[1], 0.0);
/// ```
pub fn compute_freqs(head_dim: RopeHeadDim, max_seq_len: usize, theta: RopeTheta) -> RopeFreqs {
    let pair_count = head_dim.pair_count();
    let mut freqs = Vec::with_capacity(max_seq_len * pair_count);

    for pos in 0..max_seq_len {
        for pair_idx in 0..pair_count {
            let exponent = (2 * pair_idx) as f32 / head_dim.get() as f32;
            let inverse_frequency = 1.0 / theta.get().powf(exponent);
            freqs.push(pos as f32 * inverse_frequency);
        }
    }

    RopeFreqs::from_angles(head_dim, freqs)
}

/// Applies RoPE in place to query and key vectors.
///
/// `q` and `k` are expected to contain all heads concatenated together. The
/// rotation is applied independently within each head using the validated RoPE
/// frequencies.
///
/// # Panics
///
/// Panics if:
/// - `q` and `k` have different lengths
/// - `q.len()` is not divisible by the validated `head_dim`
/// - `pos` is outside the precomputed frequency range
///
/// # Examples
///
/// ```
/// use llm_rs::math::{apply_rope, compute_freqs, RopeHeadDim, RopeTheta};
///
/// let freqs = compute_freqs(RopeHeadDim::new(2), 4, RopeTheta::new(10_000.0));
/// let mut q = vec![1.0, 0.0];
/// let mut k = vec![0.0, 1.0];
///
/// apply_rope(&mut q, &mut k, 0, &freqs);
///
/// assert_eq!(q, vec![1.0, 0.0]);
/// assert_eq!(k, vec![0.0, 1.0]);
/// ```
pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, freqs: &RopeFreqs) {
    assert_eq!(q.len(), k.len(), "q and k must have the same length");

    let head_dim = freqs.head_dim().get();
    let pair_count = freqs.head_dim().pair_count();

    assert!(
        q.len() % head_dim == 0,
        "q and k length must be divisible by head_dim"
    );

    for (q_head, k_head) in q
        .chunks_exact_mut(head_dim)
        .zip(k.chunks_exact_mut(head_dim))
    {
        for pair_idx in 0..pair_count {
            let angle = freqs.angle(pos, pair_idx);
            let (sin, cos) = angle.sin_cos();
            let even_idx = pair_idx * 2;

            let q_even = q_head[even_idx];
            let q_odd = q_head[even_idx + 1];
            q_head[even_idx] = q_even * cos - q_odd * sin;
            q_head[even_idx + 1] = q_even * sin + q_odd * cos;

            let k_even = k_head[even_idx];
            let k_odd = k_head[even_idx + 1];
            k_head[even_idx] = k_even * cos - k_odd * sin;
            k_head[even_idx + 1] = k_even * sin + k_odd * cos;
        }
    }
}
