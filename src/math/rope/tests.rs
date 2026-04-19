use super::{apply_rope, compute_freqs, RopeFreqs, RopeHeadDim, RopeTheta};

const EPSILON: f32 = 1e-5;

fn assert_slice_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());

    for (got, want) in actual.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < EPSILON,
            "expected {want}, got {got}"
        );
    }
}

#[test]
fn test_compute_freqs_small_example() {
    let freqs = compute_freqs(RopeHeadDim::new(4), 3, RopeTheta::new(10_000.0));

    let expected = [0.0, 0.0, 1.0, 0.01, 2.0, 0.02];
    assert_slice_close(freqs.as_slice(), &expected);
}

#[test]
fn test_apply_rope_pos_zero_is_identity() {
    let freqs = compute_freqs(RopeHeadDim::new(4), 2, RopeTheta::new(10_000.0));
    let mut q = vec![1.0, 2.0, 3.0, 4.0];
    let mut k = vec![5.0, 6.0, 7.0, 8.0];

    apply_rope(&mut q, &mut k, 0, &freqs);

    assert_slice_close(&q, &[1.0, 2.0, 3.0, 4.0]);
    assert_slice_close(&k, &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_apply_rope_known_ninety_degree_rotation() {
    let freqs = RopeFreqs::from_angles(
        RopeHeadDim::new(2),
        vec![0.0, std::f32::consts::FRAC_PI_2],
    );
    let mut q = vec![1.0, 0.0];
    let mut k = vec![0.0, 1.0];

    apply_rope(&mut q, &mut k, 1, &freqs);

    assert_slice_close(&q, &[0.0, 1.0]);
    assert_slice_close(&k, &[-1.0, 0.0]);
}

#[test]
fn test_apply_rope_rotates_each_head_independently() {
    let freqs = RopeFreqs::from_angles(
        RopeHeadDim::new(2),
        vec![0.0, std::f32::consts::FRAC_PI_2],
    );
    let mut q = vec![1.0, 0.0, 0.0, 1.0];
    let mut k = vec![0.0, 1.0, 1.0, 0.0];

    apply_rope(&mut q, &mut k, 1, &freqs);

    assert_slice_close(&q, &[0.0, 1.0, -1.0, 0.0]);
    assert_slice_close(&k, &[-1.0, 0.0, 0.0, 1.0]);
}
