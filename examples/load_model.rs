//! Minimal example that downloads a model and validates runtime bootstrap.

use anyhow::Result;
use lattice::InferenceRuntime;
use lattice::models::resolve_model_from_catalog;

const MODEL_CATALOG: &str = "models.toml";
const DEFAULT_MODEL: &str = "smollm-small";

#[tokio::main]
async fn main() -> Result<()> {
    let model = resolve_model_from_catalog(MODEL_CATALOG, DEFAULT_MODEL).await?;

    let runtime = InferenceRuntime::builder()
        .with_model_path(&model.path)
        .build()?;
    let status = runtime.bootstrap_status();

    println!("model-name: {}", model.name);
    println!("repo: {}", model.spec.repo);
    println!("file: {}", model.spec.file);
    println!("path: {}", model.path.display());
    println!("format: {:?}", runtime.model_format());
    println!("mapped-bytes: {}", runtime.model_bytes());
    println!("bootstrap-status: {status}");
    println!("note: this example validates download + mmap load only");

    Ok(())
}
