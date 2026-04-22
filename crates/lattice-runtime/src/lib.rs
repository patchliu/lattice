//! Runtime orchestration for the Lattice bootstrap phase.

use std::path::PathBuf;

use lattice_core::{LatticeError, Result};
use lattice_kernels::{DynamicLibraryKernelLauncher, KernelLauncher};
use lattice_model::{MappedWeights, WeightFormat};
use tracing::{debug, info};

/// Runtime configuration for a single-node CPU execution context.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Tensor alignment requested from lower layers.
    pub tensor_alignment: usize,
    /// Number of CPU threads the runtime may use.
    pub cpu_threads: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let cpu_threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);

        Self {
            tensor_alignment: 64,
            cpu_threads,
        }
    }
}

/// A bootstrap runtime that owns the mapped weights and optional external kernels.
pub struct InferenceRuntime {
    config: RuntimeConfig,
    weights: MappedWeights,
    kernel_launcher: DynamicLibraryKernelLauncher,
}

impl InferenceRuntime {
    /// Returns a builder for the runtime.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// Returns the active runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Returns the loaded weight format.
    pub fn model_format(&self) -> WeightFormat {
        self.weights.source().format()
    }

    /// Returns the mapped model size in bytes.
    pub fn model_bytes(&self) -> usize {
        self.weights.len()
    }

    /// Reports whether an external kernel library was loaded.
    pub fn has_external_kernels(&self) -> bool {
        self.kernel_launcher.is_loaded()
    }

    /// Returns a bootstrap status string for the currently initialized runtime.
    ///
    /// This reports runtime readiness only. It does not execute model inference.
    pub fn bootstrap_status(&self) -> String {
        debug!("reporting bootstrap runtime status");
        format!(
            "runtime ready: mapped {} byte(s) as {:?}",
            self.model_bytes(),
            self.model_format()
        )
    }
}

impl std::fmt::Debug for InferenceRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceRuntime")
            .field("config", &self.config)
            .field("weights_len", &self.weights.len())
            .field("weight_format", &self.weights.source().format())
            .field("has_external_kernels", &self.has_external_kernels())
            .finish()
    }
}

/// Builder for the bootstrap runtime.
#[derive(Debug, Default)]
pub struct RuntimeBuilder {
    model_path: Option<PathBuf>,
    kernel_path: Option<PathBuf>,
    config: RuntimeConfig,
}

impl RuntimeBuilder {
    /// Sets the local model path.
    pub fn with_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Sets an optional external kernel shared library.
    pub fn with_kernel_library(mut self, path: impl Into<PathBuf>) -> Self {
        self.kernel_path = Some(path.into());
        self
    }

    /// Overrides the runtime configuration.
    pub fn with_config(mut self, config: RuntimeConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds the runtime and maps the model weights.
    pub fn build(mut self) -> Result<InferenceRuntime> {
        let model_path = self
            .model_path
            .take()
            .ok_or_else(|| LatticeError::Message("model path is required".to_string()))?;

        let weights = MappedWeights::open(model_path)?;
        let mut kernel_launcher = DynamicLibraryKernelLauncher::new();

        if let Some(path) = self.kernel_path.take() {
            kernel_launcher.load(path)?;
        }

        info!(
            bytes = weights.len(),
            format = ?weights.source().format(),
            "mapped model weights"
        );

        Ok(InferenceRuntime {
            config: self.config,
            weights,
            kernel_launcher,
        })
    }
}
