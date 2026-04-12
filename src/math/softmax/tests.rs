use super::softmax;

const EPSILON: f32 = 1e-5;

#[test]
fn test_softmax_simple_input() {
    let output = softmax(&[1.0, 2.0, 3.0]);

    let expected = [0.09003057, 0.24472848, 0.66524094];
    assert_eq!(output.len(), expected.len());
    for (got, want) in output.iter().zip(expected.iter()) {
        assert!((got - want).abs() < EPSILON);
    }
}

#[test]
fn test_softmax_large_values_stability() {
    let output = softmax(&[1000.0, 1001.0, 1002.0]);

    for value in &output {
        assert!(value.is_finite());
    }

    let expected = [0.09003057, 0.24472848, 0.66524094];
    assert_eq!(output.len(), expected.len());
    for (got, want) in output.iter().zip(expected.iter()) {
        assert!((got - want).abs() < EPSILON);
    }
}

#[test]
fn test_softmax_output_sums_to_one() {
    let output = softmax(&[1.0, 2.0, 3.0]);
    let sum: f32 = output.iter().sum();

    assert!((sum - 1.0).abs() < EPSILON);
}

#[test]
#[should_panic(expected = "softmax requires all logits to be finite")]
fn test_softmax_nan_input_panics() {
    let _ = softmax(&[1.0, f32::NAN, 3.0]);
}

#[test]
#[should_panic(expected = "softmax requires all logits to be finite")]
fn test_softmax_infinite_input_panics() {
    let _ = softmax(&[1.0, f32::INFINITY, 3.0]);
}
