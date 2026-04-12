use super::{rmsnorm, RmsNormError};

const EPSILON: f32 = 1e-5;

#[test]
fn test_rmsnorm_simple_vectors() {
    let x = [1.0, 2.0, 3.0];
    let weight = [1.0, 1.0, 1.0];

    let output = rmsnorm(&x, &weight, 1e-5).unwrap();

    let expected = [0.46290955, 0.9258191, 1.3887286];
    assert_eq!(output.len(), expected.len());
    for (got, want) in output.iter().zip(expected.iter()) {
        assert!((got - want).abs() < EPSILON);
    }
}

#[test]
fn test_rmsnorm_manual_pytorch_equivalent() {
    // PyTorch-equivalent formula:
    // y = x * rsqrt(mean(x^2) + eps) * weight
    let x = [1.0, -2.0, 3.0, -4.0];
    let weight = [0.5, 1.0, 1.5, 2.0];

    let output = rmsnorm(&x, &weight, 1e-5).unwrap();

    let expected = [0.18257406, -0.73029625, 1.6431665, -2.921185];
    assert_eq!(output.len(), expected.len());
    for (got, want) in output.iter().zip(expected.iter()) {
        assert!((got - want).abs() < EPSILON);
    }
}

#[test]
fn test_rmsnorm_preserves_length() {
    let x = [0.5, -0.5, 1.5, -1.5];
    let weight = [1.0, 1.0, 1.0, 1.0];

    let output = rmsnorm(&x, &weight, 1e-5).unwrap();

    assert_eq!(output.len(), x.len());
}

#[test]
fn test_rmsnorm_mismatched_length_returns_error() {
    let x = [1.0, 2.0, 3.0];
    let weight = [1.0, 1.0];

    let err = rmsnorm(&x, &weight, 1e-5).unwrap_err();

    assert_eq!(
        err,
        RmsNormError::MismatchedLength {
            x_len: 3,
            weight_len: 2,
        }
    );
}

#[test]
fn test_rmsnorm_negative_epsilon_returns_error() {
    let x = [1.0, 2.0, 3.0];
    let weight = [1.0, 1.0, 1.0];

    let err = rmsnorm(&x, &weight, -1e-5).unwrap_err();

    assert_eq!(err, RmsNormError::NegativeEpsilon { eps: -1e-5 });
}

#[test]
fn test_rmsnorm_non_finite_denominator_returns_error() {
    let x = [f32::INFINITY, 1.0];
    let weight = [1.0, 1.0];

    let err = rmsnorm(&x, &weight, 1e-5).unwrap_err();

    assert_eq!(err, RmsNormError::NonFiniteDenominator { denom: f32::INFINITY });
}
