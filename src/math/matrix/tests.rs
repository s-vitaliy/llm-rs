use super::*;

#[test]
fn test_matrix_new_valid() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let m = Matrix::<2, 2>::new(data.clone());
    assert!(m.is_ok());
    assert_eq!(m.unwrap().data(), &data[..]);
}

#[test]
fn test_matrix_new_invalid_length() {
    let data = vec![1.0, 2.0, 3.0]; // 3 elements for a 2x2 matrix
    let m = Matrix::<2, 2>::new(data);
    assert!(m.is_err());
    let err = m.unwrap_err();
    assert_eq!(err.expected, 4);
    assert_eq!(err.actual, 3);
}

#[test]
fn test_matrix_from_slice_valid() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = Matrix::<2, 3>::from_slice(&data);
    assert!(m.is_ok());
    assert_eq!(m.unwrap().data(), &data[..]);
}

#[test]
fn test_matrix_from_slice_invalid_length() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements for a 2x3 matrix
    let m = Matrix::<2, 3>::from_slice(&data);
    assert!(m.is_err());
    let err = m.unwrap_err();
    assert_eq!(err.expected, 6);
    assert_eq!(err.actual, 5);
}

#[test]
fn test_matrix_multiplication_2x3_times_3x2() {
    // Hand-calculated test case:
    // [1 2 3] * [7  8 ] = [1*7+2*9+3*11  1*8+2*10+3*12] = [58  64]
    // [4 5 6]   [9  10]   [4*7+5*9+6*11  4*8+5*10+6*12]   [139 154]
    //           [11 12]

    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::<3, 2>::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = a * b;

    let expected = [58.0, 64.0, 139.0, 154.0];
    assert_eq!(c.data(), &expected[..]);
}

#[test]
fn test_matrix_multiplication_identity() {
    // Test that multiplying by identity returns the same matrix
    // [1 2] * [1 0] = [1 2]
    // [3 4]   [0 1]   [3 4]

    let a = Matrix::<2, 2>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let identity = Matrix::<2, 2>::from_slice(&[1.0, 0.0, 0.0, 1.0]).unwrap();
    let result = a.clone() * identity;

    assert_eq!(result.data(), a.data());
}

#[test]
fn test_matvec() {
    // Hand-calculated test case:
    // [1 2 3] * [1] = [1*1 + 2*2 + 3*3] = [14]
    // [4 5 6]   [2]   [4*1 + 5*2 + 6*3]   [32]
    //           [3]

    let m = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = [1.0, 2.0, 3.0];
    let result = m.matvec(&v);

    assert_eq!(result, [14.0, 32.0]);
}

#[test]
fn test_matvec_identity() {
    // Test that multiplying a vector by identity returns the same vector
    // [1 0] * [5] = [5]
    // [0 1]   [7]   [7]

    let identity = Matrix::<2, 2>::from_slice(&[1.0, 0.0, 0.0, 1.0]).unwrap();
    let v = [5.0, 7.0];
    let result = identity.matvec(&v);

    assert_eq!(result, v);
}

#[test]
fn test_matvec_zero() {
    // Test multiplying by zero vector
    let m = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = [0.0, 0.0, 0.0];
    let result = m.matvec(&v);

    assert_eq!(result, [0.0, 0.0]);
}
