//! Model catalog resolution backed by the Hugging Face Hub cache.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use hf_hub::api::tokio::{ApiBuilder, Progress};
use hf_hub::{Cache, Repo, RepoType};
use serde::Deserialize;

/// Describes a downloadable model artifact.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ModelSpec {
    /// Hugging Face repository identifier.
    pub repo: String,
    /// Filename inside the repository.
    pub file: String,
}

/// A manifest of named model specs.
pub type ModelCatalog = BTreeMap<String, ModelSpec>;

/// A model spec resolved to a local filesystem path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// Logical key used to select the model from the catalog.
    pub name: String,
    /// Original model spec.
    pub spec: ModelSpec,
    /// Local cached file path.
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
struct DownloadProgress {
    total_bytes: usize,
    downloaded_bytes: usize,
    started_at: Option<Instant>,
    next_report_percent: usize,
}

impl DownloadProgress {
    fn new() -> Self {
        Self {
            total_bytes: 0,
            downloaded_bytes: 0,
            started_at: None,
            next_report_percent: 10,
        }
    }
}

impl Progress for DownloadProgress {
    async fn init(&mut self, size: usize, filename: &str) {
        self.total_bytes = size;
        self.downloaded_bytes = 0;
        self.started_at = Some(Instant::now());
        self.next_report_percent = 10;

        println!(
            "download started: {filename} ({:.2} MiB)",
            size as f64 / 1024.0 / 1024.0
        );
    }

    async fn update(&mut self, size: usize) {
        self.downloaded_bytes += size;

        if self.total_bytes == 0 {
            return;
        }

        let percent = self.downloaded_bytes.saturating_mul(100) / self.total_bytes;
        if percent >= self.next_report_percent || self.downloaded_bytes == self.total_bytes {
            println!(
                "download progress: {:>3}% ({:.2}/{:.2} MiB)",
                percent.min(100),
                self.downloaded_bytes as f64 / 1024.0 / 1024.0,
                self.total_bytes as f64 / 1024.0 / 1024.0
            );

            while self.next_report_percent <= percent {
                self.next_report_percent += 10;
            }
        }
    }

    async fn finish(&mut self) {
        let elapsed = self
            .started_at
            .map(|started| started.elapsed().as_secs_f32())
            .unwrap_or_default();

        println!(
            "download finished in {:.1}s ({:.2} MiB)",
            elapsed,
            self.total_bytes as f64 / 1024.0 / 1024.0
        );
    }
}

/// Loads a model catalog from disk.
pub fn load_model_catalog(path: impl AsRef<Path>) -> Result<ModelCatalog> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read model catalog at {}", path.display()))?;
    let catalog = toml::from_str::<ModelCatalog>(&content)
        .with_context(|| format!("failed to parse model catalog at {}", path.display()))?;

    if catalog.is_empty() {
        bail!("model catalog at {} is empty", path.display());
    }

    Ok(catalog)
}

/// Resolves a model spec to a local cached path, downloading it on demand.
pub async fn resolve_model(spec: &ModelSpec) -> Result<PathBuf> {
    let repo_id = spec.repo.trim();
    let filename = spec.file.trim();

    if repo_id.is_empty() {
        bail!("repo_id must not be empty");
    }
    if filename.is_empty() {
        bail!("filename must not be empty");
    }

    println!("resolving model: {repo_id}/{filename}");

    let cache = Cache::from_env();
    let cache_repo = cache.repo(Repo::new(repo_id.to_owned(), RepoType::Model));
    if let Some(path) = cache_repo.get(filename) {
        println!("cache hit: {}", path.display());
        return Ok(path);
    }

    let api = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .context("failed to initialize Hugging Face Hub API")?;

    println!("cache miss: {}", cache.path().display());
    println!("resolving remote metadata and preparing download...");

    let path = api
        .model(repo_id.to_owned())
        .download_with_progress(filename, DownloadProgress::new())
        .await
        .with_context(|| format!("failed to fetch {repo_id}/{filename} from Hugging Face Hub"))?;

    println!("model ready: {}", path.display());
    Ok(path)
}

/// Resolves a named model from a catalog file.
pub async fn resolve_model_from_catalog(
    catalog_path: impl AsRef<Path>,
    name: &str,
) -> Result<ResolvedModel> {
    let catalog_path = catalog_path.as_ref();
    let catalog = load_model_catalog(catalog_path)?;
    let name = name.trim();

    let spec = catalog.get(name).ok_or_else(|| {
        let available = catalog.keys().cloned().collect::<Vec<_>>().join(", ");
        anyhow!(
            "model `{name}` not found in {}. available models: {available}",
            catalog_path.display()
        )
    })?;

    let path = resolve_model(spec).await?;

    Ok(ResolvedModel {
        name: name.to_owned(),
        spec: spec.clone(),
        path,
    })
}

#[cfg(test)]
mod tests {
    use super::{ModelSpec, load_model_catalog};

    #[test]
    fn parse_model_catalog() {
        let catalog =
            load_model_catalog("models.toml").expect("workspace model catalog should parse");

        let smollm = catalog
            .get("smollm-small")
            .expect("smollm-small entry should exist");

        assert_eq!(
            smollm,
            &ModelSpec {
                repo: "bartowski/SmolLM2-135M-Instruct-GGUF".to_string(),
                file: "SmolLM2-135M-Instruct-Q8_0.gguf".to_string(),
            }
        );
    }
}
