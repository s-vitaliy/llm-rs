//! Root Mean Square Layer Normalization (RMSNorm).

#[cfg(test)]
mod tests;

/// Applies RMSNorm to an input vector.
///
/// RMSNorm computes:
/// `rms = sqrt(mean(x^2) + eps)`
/// `output_i = (x_i / rms) * weight_i`
///
/// # Arguments
///
/// * `x` - Input activations.
/// * `weight` - Per-dimension scaling weights.
/// * `eps` - Small constant added for numerical stability.
///
/// # Panics
///
/// Panics if `x.len() != weight.len()`.
///
/// # Examples
///
/// ```
/// use llm_rs::math::rmsnorm;
///
/// let x = [1.0, 2.0, 3.0];
/// let weight = [1.0, 1.0, 1.0];
/// let out = rmsnorm(&x, &weight, 1e-5);
/// assert_eq!(out.len(), 3);
/// ```
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "rmsnorm requires x and weight to have the same length"
    );
    assert!(eps >= 0.0, "rmsnorm requires eps >= 0.0 (got {eps})");

    if x.is_empty() {
        return Vec::new();
    }

    let mean_square = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let denom = mean_square + eps;

    if !denom.is_finite() {
        panic!("rmsnorm: non-finite normalization term (mean_square + eps = {denom})");
    }

    if denom == 0.0 {
        // All-zero input with zero epsilon: defined behavior is to return zeros.
        return vec![0.0; x.len()];
    }

    let rms = denom.sqrt();

    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| (xi / rms) * wi)
        .collect()
}
