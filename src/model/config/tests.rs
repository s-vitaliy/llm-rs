use super::{MaxSeqLen, ModelConfig, RmsNormEps, TinyLlamaConfig};
use crate::math::RopeTheta;

const EPSILON: f32 = 1e-6;

#[test]
fn test_tiny_llama_head_dim_matches_plan() {
    assert_eq!(TinyLlamaConfig::head_dim(), 64);
}

#[test]
fn test_tiny_llama_kv_dim_matches_model_dim() {
    assert_eq!(TinyLlamaConfig::kv_dim(), 256);
}

#[test]
fn test_default_runtime_values_match_plan() {
    let config = TinyLlamaConfig::DEFAULT;

    assert!((config.rope_theta().get() - 10_000.0).abs() < EPSILON);
    assert_eq!(config.max_seq_len().get(), 512);
    assert!((config.eps().get() - 1e-5).abs() < EPSILON);
}

#[test]
fn test_new_allows_custom_runtime_values() {
    let config = TinyLlamaConfig::new(
        RopeTheta::new(5_000.0),
        MaxSeqLen::new(128),
        RmsNormEps::new(1e-6),
    );

    assert!((config.rope_theta().get() - 5_000.0).abs() < EPSILON);
    assert_eq!(config.max_seq_len().get(), 128);
    assert!((config.eps().get() - 1e-6).abs() < EPSILON);
}

#[test]
fn test_small_config_can_be_instantiated() {
    type SmallConfig = ModelConfig<1, 8, 2, 16, 32>;

    let config = SmallConfig::DEFAULT;

    assert_eq!(SmallConfig::head_dim(), 4);
    assert_eq!(SmallConfig::kv_dim(), 8);
    assert_eq!(config.max_seq_len().get(), 512);
}

#[test]
fn test_runtime_parameter_types_validate_on_construction() {
    assert_eq!(MaxSeqLen::new(32).get(), 32);
    assert!((RopeTheta::new(10_000.0).get() - 10_000.0).abs() < EPSILON);
    assert!((RmsNormEps::new(1e-5).get() - 1e-5).abs() < EPSILON);
}
