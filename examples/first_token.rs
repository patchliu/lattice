//! First-token example that accepts raw token ids and returns the final prefill argmax token.

use anyhow::{Result, anyhow};
use lattice::InferenceRuntime;
use lattice::models::resolve_model_from_catalog;

const MODEL_CATALOG: &str = "models.toml";
const DEFAULT_MODEL: &str = "smollm-small";
const DEFAULT_TOKEN_IDS: &str = "1,2,3";

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let selected_model = args.next().unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let token_ids = parse_token_ids(args.next().as_deref().unwrap_or(DEFAULT_TOKEN_IDS))?;
    let model = resolve_model_from_catalog(MODEL_CATALOG, &selected_model).await?;

    let runtime = InferenceRuntime::builder()
        .with_model_path(&model.path)
        .build()?;
    let loaded = runtime.load_llama()?;
    let output = loaded.prefill(&token_ids)?;

    println!("model-name: {}", model.name);
    println!("path: {}", model.path.display());
    println!("prompt-token-ids: {:?}", token_ids);
    println!("next-token-id: {}", output.next_token_id);
    println!("logits-len: {}", output.logits.len());
    println!("max-logit: {:.6}", output.logits[output.next_token_id as usize]);

    Ok(())
}

fn parse_token_ids(input: &str) -> Result<Vec<u32>> {
    let token_ids = input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|error| anyhow!("invalid token id `{value}`: {error}"))
        })
        .collect::<Result<Vec<_>>>()?;

    if token_ids.is_empty() {
        return Err(anyhow!("at least one token id is required"));
    }

    Ok(token_ids)
}
