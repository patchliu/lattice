//! Minimal example that downloads a LLaMA-family GGUF and validates runtime bindings.

use anyhow::Result;
use lattice::InferenceRuntime;
use lattice::models::resolve_model_from_catalog;

const MODEL_CATALOG: &str = "models.toml";
const DEFAULT_MODEL: &str = "llama3-basic";

#[tokio::main]
async fn main() -> Result<()> {
    let selected_model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let model = resolve_model_from_catalog(MODEL_CATALOG, &selected_model).await?;

    let runtime = InferenceRuntime::builder()
        .with_model_path(&model.path)
        .build()?;
    let status = runtime.bootstrap_status();
    let loaded = runtime.load_llama()?;
    let gguf = loaded.gguf();
    let llama = loaded.model();
    let first_block = llama
        .tensors
        .block(0)
        .expect("llama model should have at least one transformer block");

    println!("model-name: {}", model.name);
    println!("repo: {}", model.spec.repo);
    println!("file: {}", model.spec.file);
    println!("path: {}", model.path.display());
    println!("format: {:?}", runtime.model_format());
    println!("mapped-bytes: {}", runtime.model_bytes());
    println!(
        "gguf-architecture: {}",
        gguf.metadata().architecture_name().unwrap_or("unknown")
    );
    println!(
        "llama-model-name: {}",
        llama.spec.model_name.as_deref().unwrap_or("<unnamed>")
    );
    println!("llama-context-length: {}", llama.spec.context_length);
    println!("llama-embedding-length: {}", llama.spec.embedding_length);
    println!("llama-block-count: {}", llama.spec.block_count);
    println!("llama-head-count: {}", llama.spec.attention_head_count);
    println!(
        "llama-head-count-kv: {}",
        llama.spec.attention_head_count_kv
    );
    println!("llama-vocab-size: {}", llama.spec.vocab_size);
    println!(
        "tensor-token-embd: {} {:?}",
        llama.tensors.token_embeddings.name(),
        llama.tensors.token_embeddings.dimensions()
    );
    println!(
        "tensor-block0-attn-q: {} {:?}",
        first_block.attention_q.name(),
        first_block.attention_q.dimensions()
    );
    println!(
        "tensor-output: {} {:?}",
        llama.tensors.output.name(),
        llama.tensors.output.dimensions()
    );
    println!("llama-output-tied: {}", llama.tensors.output_is_tied);
    println!("bootstrap-status: {status}");
    println!("note: this example validates download + mmap load + runtime llama binding only");

    Ok(())
}
