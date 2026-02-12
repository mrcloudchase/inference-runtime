use crate::error::{Result, TensorError};
use std::fmt;

/// A tensor shape, wrapping a vector of dimension sizes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from a vector of dimensions.
    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    /// Create a shape from a slice of dimensions.
    pub fn from_slice(dims: &[usize]) -> Self {
        Shape {
            dims: dims.to_vec(),
        }
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements (product of all dimension sizes).
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns the size of dimension `i`.
    ///
    /// # Panics
    /// Panics if `i >= ndim()`.
    pub fn dim(&self, i: usize) -> usize {
        self.dims[i]
    }

    /// Returns a reference to the underlying dimension sizes.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Computes row-major contiguous strides for this shape.
    ///
    /// For a shape [d0, d1, d2], the strides are [d1*d2, d2, 1].
    pub fn strides(&self) -> Vec<usize> {
        if self.dims.is_empty() {
            return vec![];
        }
        let mut strides = vec![0usize; self.dims.len()];
        strides[self.dims.len() - 1] = 1;
        for i in (0..self.dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Checks if the given strides correspond to a contiguous (row-major) layout
    /// for this shape.
    pub fn is_contiguous(&self, strides: &[usize]) -> bool {
        if strides.len() != self.dims.len() {
            return false;
        }
        let expected = self.strides();
        strides == expected.as_slice()
    }

    /// Compute the broadcast shape of `a` and `b` using numpy-style broadcasting rules.
    ///
    /// Rules:
    /// 1. If the shapes have different numbers of dimensions, the shorter shape is
    ///    padded with ones on the left.
    /// 2. For each dimension, sizes must either be equal, or one of them must be 1.
    ///    The output dimension is the maximum of the two.
    pub fn broadcast_shape(a: &Shape, b: &Shape) -> Result<Shape> {
        let max_ndim = a.ndim().max(b.ndim());
        let mut result = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            // Index from the right: dim at position (ndim - 1 - i) from the right
            let da = if i < a.ndim() {
                a.dims[a.ndim() - 1 - i]
            } else {
                1
            };
            let db = if i < b.ndim() {
                b.dims[b.ndim() - 1 - i]
            } else {
                1
            };

            if da == db {
                result.push(da);
            } else if da == 1 {
                result.push(db);
            } else if db == 1 {
                result.push(da);
            } else {
                return Err(TensorError::BroadcastError {
                    a: a.dims.clone(),
                    b: b.dims.clone(),
                });
            }
        }

        result.reverse();
        Ok(Shape::new(result))
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::from_slice(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_shape() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dim(0), 2);
        assert_eq!(s.dim(1), 3);
        assert_eq!(s.dim(2), 4);
    }

    #[test]
    fn test_strides() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_is_contiguous() {
        let s = Shape::new(vec![2, 3, 4]);
        assert!(s.is_contiguous(&[12, 4, 1]));
        assert!(!s.is_contiguous(&[12, 1, 4]));
    }

    #[test]
    fn test_scalar_shape() {
        let s = Shape::new(vec![]);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1); // product of empty = 1
        assert_eq!(s.strides(), vec![]);
    }

    #[test]
    fn test_broadcast_same() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![2, 3]);
        let c = Shape::broadcast_shape(&a, &b).unwrap();
        assert_eq!(c.dims(), &[2, 3]);
    }

    #[test]
    fn test_broadcast_expand() {
        let a = Shape::new(vec![2, 1]);
        let b = Shape::new(vec![1, 3]);
        let c = Shape::broadcast_shape(&a, &b).unwrap();
        assert_eq!(c.dims(), &[2, 3]);
    }

    #[test]
    fn test_broadcast_different_ndim() {
        let a = Shape::new(vec![3]);
        let b = Shape::new(vec![2, 3]);
        let c = Shape::broadcast_shape(&a, &b).unwrap();
        assert_eq!(c.dims(), &[2, 3]);
    }

    #[test]
    fn test_broadcast_error() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![2, 4]);
        assert!(Shape::broadcast_shape(&a, &b).is_err());
    }
}
