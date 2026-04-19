//! Strongly typed model weights.
//!
//! The matrix shapes are encoded in the type so attention and FFN weights keep
//! the expected dimensions at compile time.

use crate::math::Matrix;

#[cfg(test)]
mod tests;

/// Attention projection weights for one transformer layer.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionWeights<const D: usize> {
    pub wq: Matrix<D, D>,
    pub wk: Matrix<D, D>,
    pub wv: Matrix<D, D>,
    pub wo: Matrix<D, D>,
}

/// Feed-forward weights for one transformer layer.
///
/// The shapes match the current `Matrix::matvec` convention:
/// `Matrix<ROWS, COLS> * [COLS] -> [ROWS]`.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnWeights<const D: usize, const F: usize> {
    pub w1: Matrix<F, D>,
    pub w2: Matrix<F, D>,
    pub w3: Matrix<D, F>,
}

/// All weights required for one transformer layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerWeights<const D: usize, const F: usize> {
    pub attention: AttentionWeights<D>,
    pub ffn: FfnWeights<D, F>,
    pub attention_norm: [f32; D],
    pub ffn_norm: [f32; D],
}
