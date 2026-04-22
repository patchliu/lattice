//! Command-line entrypoint for bootstrapping the Lattice workspace.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use lattice_runtime::InferenceRuntime;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "lattice",
    version,
    about = "CPU-first Rust LLM inference runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Prints environment and workspace bootstrap status.
    Doctor,
    /// Runs the bootstrap L1 flow against a local model path.
    Run {
        /// Path to a local GGUF or Safetensors model file.
        #[arg(long)]
        model: PathBuf,
        /// Prompt used for the placeholder first-token path.
        #[arg(long)]
        prompt: String,
        /// Optional path to a shared object that exports optimized kernels.
        #[arg(long)]
        kernels: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();
    match cli.command {
        Commands::Doctor => doctor(),
        Commands::Run {
            model,
            prompt,
            kernels,
        } => run(model, prompt, kernels)?,
    }

    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("lattice_runtime=info"));

    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn doctor() {
    println!("lattice workspace initialized");
    println!(
        "next step: cargo run -p lattice-cli -- run --model /path/to/model.gguf --prompt \"hello\""
    );
}

fn run(model: PathBuf, prompt: String, kernels: Option<PathBuf>) -> Result<()> {
    let mut builder = InferenceRuntime::builder().with_model_path(model);
    if let Some(kernels) = kernels {
        builder = builder.with_kernel_library(kernels);
    }

    let runtime = builder.build()?;
    let token = runtime.generate_first_token(&prompt)?;

    println!("format: {:?}", runtime.model_format());
    println!("mapped-bytes: {}", runtime.model_bytes());
    println!("external-kernels: {}", runtime.has_external_kernels());
    println!("first-token: {token}");

    Ok(())
}
