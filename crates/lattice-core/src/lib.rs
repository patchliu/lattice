//! Core shared types for the Lattice workspace.

use std::{fmt, path::PathBuf};

/// Convenient result type used across workspace crates.
pub type Result<T> = std::result::Result<T, LatticeError>;

/// Base error type shared by workspace crates.
#[derive(Debug, thiserror::Error)]
pub enum LatticeError {
    /// Returned when the weight file format is not supported.
    #[error("unsupported model format for path: {0}")]
    UnsupportedModelFormat(PathBuf),
    /// Wraps standard I/O failures.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A generic message error for early-stage bootstrap code.
    #[error("{0}")]
    Message(String),
}

/// A stable request identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(u64);

impl RequestId {
    /// Creates a new request identifier.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw identifier value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "req-{}", self.0)
    }
}
