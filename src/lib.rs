//! Public facade crate for workspace-level examples.

pub mod models;

pub use lattice_runtime::InferenceRuntime;
pub use models::{
    ModelCatalog, ModelSpec, ResolvedModel, load_model_catalog, resolve_model,
    resolve_model_from_catalog,
};

/// Re-export model-layer types behind a stable namespace.
pub mod model {
    pub use lattice_model::{MappedWeights, ModelSource, WeightFormat};

    /// GGUF parsing and architecture metadata helpers.
    pub mod gguf {
        pub use lattice_model::gguf::*;
    }
}
