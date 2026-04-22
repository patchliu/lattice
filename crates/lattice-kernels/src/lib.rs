//! Dynamic kernel loading primitives.

use std::path::{Path, PathBuf};

use lattice_core::{LatticeError, Result};

/// Interface used to load precompiled external kernels.
pub trait KernelLauncher {
    /// Loads a shared library from the provided path.
    fn load(&mut self, library_path: impl AsRef<Path>) -> Result<()>;

    /// Returns whether a library has been loaded.
    fn is_loaded(&self) -> bool;

    /// Returns the loaded library path if present.
    fn loaded_library(&self) -> Option<&Path>;
}

/// Lightweight launcher backed by `libloading`.
#[derive(Default)]
pub struct DynamicLibraryKernelLauncher {
    library_path: Option<PathBuf>,
    handle: Option<libloading::Library>,
}

impl DynamicLibraryKernelLauncher {
    /// Creates a new unloaded launcher.
    pub fn new() -> Self {
        Self::default()
    }
}

impl KernelLauncher for DynamicLibraryKernelLauncher {
    fn load(&mut self, library_path: impl AsRef<Path>) -> Result<()> {
        let path = library_path.as_ref().to_path_buf();
        // SAFETY: The dynamic library handle is stored in `self.handle` and therefore
        // lives for as long as the launcher instance.
        let handle = unsafe { libloading::Library::new(&path) }.map_err(|error| {
            LatticeError::Message(format!(
                "failed to load kernel library {}: {error}",
                path.display()
            ))
        })?;

        self.library_path = Some(path);
        self.handle = Some(handle);
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.handle.is_some()
    }

    fn loaded_library(&self) -> Option<&Path> {
        self.library_path.as_deref()
    }
}

impl std::fmt::Debug for DynamicLibraryKernelLauncher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicLibraryKernelLauncher")
            .field("library_path", &self.library_path)
            .field("is_loaded", &self.is_loaded())
            .finish()
    }
}
