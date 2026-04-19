//! Mathematical operations for LLM inference.
//!
//! This module provides core mathematical primitives including matrix operations.

mod matrix;
mod rmsnorm;
mod rope;
mod softmax;

pub use matrix::{Matrix, ShapeError};
pub use rmsnorm::{rmsnorm, RmsNormError};
pub use rope::{apply_rope, compute_freqs, RopeFreqs, RopeHeadDim, RopeTheta};
pub use softmax::softmax;
