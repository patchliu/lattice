//! Public facade crate for workspace-level examples.

pub mod models;

pub use lattice_runtime::InferenceRuntime;
pub use models::{
    ModelCatalog, ModelSpec, ResolvedModel, load_model_catalog, resolve_model,
    resolve_model_from_catalog,
};
