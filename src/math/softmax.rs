//! Numerically stable softmax.

#[cfg(test)]
mod tests;

/// Computes softmax probabilities from logits using a numerically stable formulation.
///
/// This implementation subtracts the maximum logit before exponentiation to avoid
/// overflow when logits contain large values.
///
/// # Arguments
///
/// * `logits` - Input logits.
///
/// # Returns
///
/// A vector of probabilities with the same length as `logits`. If `logits` is empty,
/// returns an empty vector.
///
/// # Panics
///
/// Panics if any input logit is non-finite, or if the normalization term becomes
/// non-finite or non-positive.
///
/// # Examples
///
/// ```
/// use llm_rs::math::softmax;
///
/// let probs = softmax(&[1.0, 2.0, 3.0]);
/// let sum: f32 = probs.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
/// ```
///
/// An empty logits slice returns an empty probability vector:
///
/// ```
/// use llm_rs::math::softmax;
///
/// let logits: [f32; 0] = [];
/// let probs = softmax(&logits);
/// assert!(probs.is_empty());
/// ```
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    if logits.iter().any(|x| !x.is_finite()) {
        panic!("softmax requires all logits to be finite");
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if !max_logit.is_finite() {
        panic!("softmax requires a finite maximum logit");
    }

    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let exp_sum: f32 = exp_values.iter().sum();

    if !exp_sum.is_finite() || exp_sum <= 0.0 {
        panic!("softmax produced an invalid normalization term: {exp_sum}");
    }

    exp_values.iter().map(|&x| x / exp_sum).collect()
}
