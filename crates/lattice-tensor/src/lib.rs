//! Tensor structures backed by aligned CPU memory.

use lattice_allocator::{AlignedBuffer, BufferAllocator, CpuAllocator};

/// Scalar element types supported by the bootstrap runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating-point.
    F32,
    /// 16-bit floating-point placeholder.
    F16,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
}

impl DType {
    /// Returns the byte width of the dtype.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

/// Tensor dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape(Vec<usize>);

impl TensorShape {
    /// Creates a tensor shape from the provided dimensions.
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    /// Returns the raw dimension slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Returns the total number of logical elements.
    pub fn element_count(&self) -> usize {
        self.0.iter().product()
    }
}

/// A CPU tensor backed by an aligned byte buffer.
#[derive(Debug)]
pub struct CpuTensor {
    shape: TensorShape,
    dtype: DType,
    buffer: AlignedBuffer,
}

impl CpuTensor {
    /// Allocates a zero-initialized tensor.
    pub fn zeros(shape: TensorShape, dtype: DType, alignment: usize) -> Self {
        let bytes = shape.element_count().max(1) * dtype.size_in_bytes();
        let allocator = CpuAllocator;
        let buffer = allocator.allocate(bytes, alignment);

        Self {
            shape,
            dtype,
            buffer,
        }
    }

    /// Returns the tensor shape.
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns the tensor dtype.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the tensor size in bytes.
    pub fn len_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the backing bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    /// Returns the backing bytes mutably.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::{CpuTensor, DType, TensorShape};

    #[test]
    fn zero_tensor_allocates_expected_bytes() {
        let tensor = CpuTensor::zeros(TensorShape::new([2, 4, 8]), DType::F32, 64);
        assert_eq!(tensor.len_bytes(), 2 * 4 * 8 * 4);
    }
}
