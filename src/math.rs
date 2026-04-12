//! Mathematical operations for LLM inference.
//!
//! This module provides core mathematical primitives including matrix operations.

mod matrix;
mod rmsnorm;
mod softmax;

pub use matrix::{Matrix, ShapeError};
pub use rmsnorm::{rmsnorm, RmsNormError};
pub use softmax::softmax;
