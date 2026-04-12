//! Root Mean Square Layer Normalization (RMSNorm).

use std::fmt;

#[cfg(test)]
mod tests;

/// Error type for invalid RMSNorm inputs or intermediate values.
#[derive(Debug, Clone, PartialEq)]
pub enum RmsNormError {
    MismatchedLength { x_len: usize, weight_len: usize },
    NegativeEpsilon { eps: f32 },
    NonFiniteDenominator { denom: f32 },
}

impl fmt::Display for RmsNormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedLength { x_len, weight_len } => {
                write!(
                    f,
                    "x and weight must have the same length (got x={}, weight={})",
                    x_len, weight_len
                )
            }
            Self::NegativeEpsilon { eps } => {
                write!(f, "eps must be >= 0.0 (got {eps})")
            }
            Self::NonFiniteDenominator { denom } => {
                write!(f, "normalization term must be finite (got {denom})")
            }
        }
    }
}

impl std::error::Error for RmsNormError {}

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
/// # Errors
///
/// Returns an error if:
/// - `x.len() != weight.len()`
/// - `eps < 0.0`
/// - `mean_square + eps` is not finite
///
/// # Examples
///
/// ```
/// use llm_rs::math::rmsnorm;
///
/// let x = [1.0, 2.0, 3.0];
/// let weight = [1.0, 1.0, 1.0];
/// let out = rmsnorm(&x, &weight, 1e-5).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, RmsNormError> {
    if x.len() != weight.len() {
        return Err(RmsNormError::MismatchedLength {
            x_len: x.len(),
            weight_len: weight.len(),
        });
    }

    if eps < 0.0 {
        return Err(RmsNormError::NegativeEpsilon { eps });
    }

    if x.is_empty() {
        return Ok(Vec::new());
    }

    let mean_square = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let denom = mean_square + eps;

    if !denom.is_finite() {
        return Err(RmsNormError::NonFiniteDenominator { denom });
    }

    if denom == 0.0 {
        // All-zero input with zero epsilon: defined behavior is to return zeros.
        return Ok(vec![0.0; x.len()]);
    }

    let rms = denom.sqrt();

    Ok(x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| (xi / rms) * wi)
        .collect())
}
