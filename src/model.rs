//! Model definitions and components.
//!
//! This module contains the model architecture and configuration.

mod config;
mod weights;

pub use config::{MaxSeqLen, ModelConfig, RmsNormEps, TinyLlamaConfig};
pub use weights::{AttentionWeights, FfnWeights, LayerWeights};
