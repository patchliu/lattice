//! Aligned CPU allocation primitives.

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::ptr::NonNull;

/// Abstraction over aligned buffer allocation.
pub trait BufferAllocator {
    /// Allocates a zero-initialized buffer with the requested size and alignment.
    fn allocate(&self, len: usize, alignment: usize) -> AlignedBuffer;
}

/// Default CPU allocator used during bootstrap.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuAllocator;

impl BufferAllocator for CpuAllocator {
    fn allocate(&self, len: usize, alignment: usize) -> AlignedBuffer {
        AlignedBuffer::new(len, alignment)
    }
}

/// A manually aligned byte buffer suitable for tensor storage.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    /// Allocates a zero-initialized buffer.
    pub fn new(len: usize, alignment: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "alignment must be a power of two"
        );

        let len = len.max(1);
        let layout = Layout::from_size_align(len, alignment)
            .expect("aligned buffer layout must be constructible");

        // SAFETY: `layout` was constructed above and is valid.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));

        Self { ptr, len, layout }
    }

    /// Returns the buffer length in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the buffer is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the configured alignment.
    pub const fn alignment(&self) -> usize {
        self.layout.align()
    }

    /// Exposes the buffer as an immutable byte slice.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` points to an allocation of `len` bytes owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Exposes the buffer as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` points to an allocation of `len` bytes owned by `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len)
            .field("alignment", &self.alignment())
            .finish()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated with `alloc_zeroed` using the same layout.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::{AlignedBuffer, BufferAllocator, CpuAllocator};

    #[test]
    fn aligned_buffer_reports_alignment() {
        let buffer = AlignedBuffer::new(128, 64);
        assert_eq!(buffer.alignment(), 64);
        assert_eq!(buffer.len(), 128);
    }

    #[test]
    fn cpu_allocator_allocates_buffer() {
        let allocator = CpuAllocator;
        let buffer = allocator.allocate(256, 32);
        assert_eq!(buffer.len(), 256);
        assert_eq!(buffer.alignment(), 32);
    }
}
