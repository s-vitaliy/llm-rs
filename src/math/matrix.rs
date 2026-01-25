//! Matrix type with compile-time shape checking using const generics.

use std::fmt;
use std::ops::Mul;

/// Error type for matrix shape mismatches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeError {
    expected: usize,
    actual: usize,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Shape error: expected {} elements, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for ShapeError {}

/// A matrix with compile-time dimensions.
///
/// The matrix dimensions are encoded in the type using const generics,
/// allowing the compiler to verify dimension compatibility at compile time.
///
/// # Type Parameters
///
/// * `ROWS` - Number of rows in the matrix
/// * `COLS` - Number of columns in the matrix
///
/// # Examples
///
/// ```
/// use llm_rs::math::Matrix;
///
/// // Create a 2x3 matrix
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let m = Matrix::<2, 3>::new(data).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
    data: Vec<f32>,
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Creates a new matrix from a vector of data.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector containing the matrix elements in row-major order
    ///
    /// # Errors
    ///
    /// Returns a `ShapeError` if the length of `data` doesn't match `ROWS * COLS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_rs::math::Matrix;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let m = Matrix::<2, 2>::new(data).unwrap();
    /// ```
    pub fn new(data: Vec<f32>) -> Result<Self, ShapeError> {
        let expected = ROWS * COLS;
        let actual = data.len();
        if actual != expected {
            return Err(ShapeError { expected, actual });
        }
        Ok(Matrix { data })
    }

    /// Creates a new matrix from a slice of data.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice containing the matrix elements in row-major order
    ///
    /// # Errors
    ///
    /// Returns a `ShapeError` if the length of `data` doesn't match `ROWS * COLS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_rs::math::Matrix;
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let m = Matrix::<2, 3>::from_slice(&data).unwrap();
    /// ```
    pub fn from_slice(data: &[f32]) -> Result<Self, ShapeError> {
        let expected = ROWS * COLS;
        let actual = data.len();
        if actual != expected {
            return Err(ShapeError { expected, actual });
        }
        Ok(Matrix {
            data: data.to_vec(),
        })
    }

    /// Multiplies this matrix by a column vector.
    ///
    /// Computes the matrix-vector product: `result = self * vec`
    ///
    /// # Arguments
    ///
    /// * `vec` - A column vector with length `COLS`
    ///
    /// # Returns
    ///
    /// A result vector with length `ROWS`
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_rs::math::Matrix;
    ///
    /// let m = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let v = [1.0, 2.0, 3.0];
    /// let result = m.matvec(&v);
    /// assert_eq!(result, [14.0, 32.0]); // [1*1+2*2+3*3, 4*1+5*2+6*3]
    /// ```
    pub fn matvec(&self, vec: &[f32; COLS]) -> [f32; ROWS] {
        let mut result = [0.0f32; ROWS];
        for i in 0..ROWS {
            let mut sum = 0.0;
            for j in 0..COLS {
                sum += self.data[i * COLS + j] * vec[j];
            }
            result[i] = sum;
        }
        result
    }

    /// Returns a reference to the underlying data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

/// Matrix multiplication with compile-time dimension checking.
///
/// Multiplies a `ROWS x K` matrix by a `K x COLS` matrix to produce a `ROWS x COLS` matrix.
/// The dimension `K` must match at compile time, otherwise the code won't compile.
///
/// # Type Parameters
///
/// * `ROWS` - Number of rows in the left matrix and result
/// * `K` - Number of columns in the left matrix and rows in the right matrix
/// * `COLS` - Number of columns in the right matrix and result
///
/// # Examples
///
/// ```
/// use llm_rs::math::Matrix;
///
/// let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let b = Matrix::<3, 2>::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
/// let c = a * b; // Results in Matrix<2, 2>
/// ```
///
/// This won't compile due to dimension mismatch:
/// ```compile_fail
/// use llm_rs::math::Matrix;
///
/// let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let b = Matrix::<2, 2>::from_slice(&[7.0, 8.0, 9.0, 10.0]).unwrap();
/// let c = a * b; // ERROR: dimension mismatch (3 != 2)
/// ```
impl<const ROWS: usize, const K: usize, const COLS: usize> Mul<Matrix<K, COLS>>
    for Matrix<ROWS, K>
{
    type Output = Matrix<ROWS, COLS>;

    fn mul(self, rhs: Matrix<K, COLS>) -> Self::Output {
        let mut result = vec![0.0f32; ROWS * COLS];

        for i in 0..ROWS {
            for j in 0..COLS {
                let mut sum = 0.0;
                for k in 0..K {
                    sum += self.data[i * K + k] * rhs.data[k * COLS + j];
                }
                result[i * COLS + j] = sum;
            }
        }

        Matrix { data: result }
    }
}

#[cfg(test)]
mod tests {
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
}
