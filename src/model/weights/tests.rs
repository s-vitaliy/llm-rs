use super::{AttentionWeights, FfnWeights, LayerWeights};
use crate::math::Matrix;

fn matrix_with_value<const ROWS: usize, const COLS: usize>(value: f32) -> Matrix<ROWS, COLS> {
    Matrix::new(vec![value; ROWS * COLS]).unwrap()
}

#[test]
fn test_attention_weights_can_be_instantiated() {
    let weights = AttentionWeights::<4> {
        wq: matrix_with_value(1.0),
        wk: matrix_with_value(2.0),
        wv: matrix_with_value(3.0),
        wo: matrix_with_value(4.0),
    };

    assert_eq!(weights.wq.data().len(), 16);
    assert_eq!(weights.wk.data().len(), 16);
    assert_eq!(weights.wv.data().len(), 16);
    assert_eq!(weights.wo.data().len(), 16);
}

#[test]
fn test_ffn_weights_match_matvec_orientation() {
    let weights = FfnWeights::<4, 8> {
        w1: matrix_with_value(1.0),
        w2: matrix_with_value(2.0),
        w3: matrix_with_value(3.0),
    };

    assert_eq!(weights.w1.data().len(), 32);
    assert_eq!(weights.w2.data().len(), 32);
    assert_eq!(weights.w3.data().len(), 32);
}

#[test]
fn test_layer_weights_bundle_all_components() {
    let layer = LayerWeights::<4, 8> {
        attention: AttentionWeights {
            wq: matrix_with_value(1.0),
            wk: matrix_with_value(2.0),
            wv: matrix_with_value(3.0),
            wo: matrix_with_value(4.0),
        },
        ffn: FfnWeights {
            w1: matrix_with_value(5.0),
            w2: matrix_with_value(6.0),
            w3: matrix_with_value(7.0),
        },
        attention_norm: [1.0, 1.0, 1.0, 1.0],
        ffn_norm: [0.5, 0.5, 0.5, 0.5],
    };

    assert_eq!(layer.attention_norm, [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(layer.ffn_norm, [0.5, 0.5, 0.5, 0.5]);
    assert_eq!(layer.ffn.w1.data().len(), 32);
    assert_eq!(layer.ffn.w3.data().len(), 32);
}
