//! Runtime orchestration for the Lattice bootstrap phase.

use std::path::PathBuf;

use lattice_core::{LatticeError, Result};
use lattice_kernels::{DynamicLibraryKernelLauncher, KernelLauncher};
use lattice_model::{GgufFile, LlamaModel, MappedWeights, WeightFormat};
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

/// A parsed, architecture-specific LLaMA model view backed by runtime-owned weights.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedLlama<'a> {
    gguf: GgufFile<'a>,
    model: LlamaModel<'a>,
}

impl<'a> LoadedLlama<'a> {
    /// Returns the parsed GGUF file view.
    pub fn gguf(&self) -> &GgufFile<'a> {
        &self.gguf
    }

    /// Returns the validated LLaMA binding.
    pub fn model(&self) -> &LlamaModel<'a> {
        &self.model
    }
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

    /// Parses the mapped GGUF weights as a runtime-ready LLaMA model.
    pub fn load_llama(&self) -> Result<LoadedLlama<'_>> {
        let gguf = self.weights.gguf_file()?;
        let model = LlamaModel::from_gguf(&gguf)?;

        Ok(LoadedLlama { gguf, model })
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::InferenceRuntime;

    #[test]
    fn loads_llama_from_runtime_weights() {
        let path = write_temp_file("runtime-loads-llama", &build_llama_gguf(None));
        let runtime = InferenceRuntime::builder()
            .with_model_path(&path)
            .build()
            .expect("runtime should build");

        let loaded = runtime.load_llama().expect("llama model should load");
        assert_eq!(loaded.model().spec.model_name.as_deref(), Some("TinyLlama"));
        assert!(loaded.model().tensors.output_is_tied);
        assert_eq!(loaded.model().tensors.output.name(), "token_embd.weight");
        assert_eq!(loaded.gguf().metadata().architecture_name(), Some("llama"));

        fs::remove_file(path).expect("temp file should be removed");
    }

    fn write_temp_file(prefix: &str, bytes: &[u8]) -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        path.push(format!("{prefix}-{unique}.gguf"));
        fs::write(&path, bytes).expect("temp GGUF should be written");
        path
    }

    fn build_llama_gguf(output_name: Option<&str>) -> Vec<u8> {
        let embedding = 8_usize;
        let feed_forward = 16_usize;
        let vocab = 32_usize;

        let metadata = vec![
            MetadataEntry::string("general.architecture", "llama"),
            MetadataEntry::string("general.name", "TinyLlama"),
            MetadataEntry::u32("general.alignment", 32),
            MetadataEntry::u32("llama.context_length", 128),
            MetadataEntry::u32("llama.embedding_length", embedding as u32),
            MetadataEntry::u32("llama.block_count", 1),
            MetadataEntry::u32("llama.feed_forward_length", feed_forward as u32),
            MetadataEntry::u32("llama.attention.head_count", 2),
            MetadataEntry::array_string(
                "tokenizer.ggml.tokens",
                (0..vocab)
                    .map(|index| format!("tok-{index}"))
                    .collect::<Vec<_>>(),
            ),
        ];

        let mut tensors = vec![
            tensor("token_embd.weight", vec![embedding as u64, vocab as u64]),
            tensor("output_norm.weight", vec![embedding as u64]),
        ];
        if let Some(output_name) = output_name {
            tensors.push(tensor(output_name, vec![embedding as u64, vocab as u64]));
        }

        for block in 0..1 {
            tensors.push(tensor(
                &format!("blk.{block}.attn_norm.weight"),
                vec![embedding as u64],
            ));
            tensors.push(tensor(
                &format!("blk.{block}.attn_q.weight"),
                vec![embedding as u64, embedding as u64],
            ));
            tensors.push(tensor(
                &format!("blk.{block}.attn_k.weight"),
                vec![embedding as u64, embedding as u64],
            ));
            tensors.push(tensor(
                &format!("blk.{block}.attn_v.weight"),
                vec![embedding as u64, embedding as u64],
            ));
            tensors.push(tensor(
                &format!("blk.{block}.attn_output.weight"),
                vec![embedding as u64, embedding as u64],
            ));
            tensors.push(tensor(
                &format!("blk.{block}.ffn_norm.weight"),
                vec![embedding as u64],
            ));
            tensors.push(tensor(
                &format!("blk.{block}.ffn_gate.weight"),
                vec![embedding as u64, feed_forward as u64],
            ));
            tensors.push(tensor(
                &format!("blk.{block}.ffn_down.weight"),
                vec![feed_forward as u64, embedding as u64],
            ));
            tensors.push(tensor(
                &format!("blk.{block}.ffn_up.weight"),
                vec![embedding as u64, feed_forward as u64],
            ));
        }

        build_gguf(&metadata, &tensors, 32)
    }

    fn tensor(name: &str, dimensions: Vec<u64>) -> TensorEntry {
        let byte_len = lattice_model::GgmlType::F32
            .tensor_byte_len(&dimensions)
            .expect("tensor byte length should fit");
        TensorEntry {
            name: name.to_string(),
            dimensions,
            ggml_type: lattice_model::GgmlType::F32.raw(),
            data: vec![0_u8; byte_len],
        }
    }

    fn build_gguf(
        metadata_entries: &[MetadataEntry],
        tensor_entries: &[TensorEntry],
        alignment: usize,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, tensor_entries.len() as u64);
        push_u64(&mut bytes, metadata_entries.len() as u64);

        for entry in metadata_entries {
            push_string(&mut bytes, &entry.key);
            push_value(&mut bytes, &entry.value);
        }

        let mut next_offset = 0_usize;
        let mut aligned_offsets = Vec::with_capacity(tensor_entries.len());
        for tensor in tensor_entries {
            next_offset = align_to(next_offset, alignment);
            aligned_offsets.push(next_offset as u64);
            next_offset += tensor.data.len();
        }

        for (tensor, offset) in tensor_entries.iter().zip(aligned_offsets.iter()) {
            push_string(&mut bytes, &tensor.name);
            push_u32(&mut bytes, tensor.dimensions.len() as u32);
            for dimension in &tensor.dimensions {
                push_u64(&mut bytes, *dimension);
            }
            push_u32(&mut bytes, tensor.ggml_type);
            push_u64(&mut bytes, *offset);
        }

        let aligned_data_start = align_to(bytes.len(), alignment);
        bytes.resize(aligned_data_start, 0);

        let mut data_section = vec![0_u8; next_offset];
        for (tensor, offset) in tensor_entries.iter().zip(aligned_offsets.iter()) {
            let start = *offset as usize;
            let end = start + tensor.data.len();
            data_section[start..end].copy_from_slice(&tensor.data);
        }

        bytes.extend_from_slice(&data_section);
        bytes
    }

    fn align_to(value: usize, alignment: usize) -> usize {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + (alignment - remainder)
        }
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_value(bytes: &mut Vec<u8>, value: &TestValue) {
        match value {
            TestValue::U32(value) => {
                push_u32(bytes, 4);
                push_u32(bytes, *value);
            }
            TestValue::String(value) => {
                push_u32(bytes, 8);
                push_string(bytes, value);
            }
            TestValue::ArrayString(values) => {
                push_u32(bytes, 9);
                push_u32(bytes, 8);
                push_u64(bytes, values.len() as u64);
                for value in values {
                    push_string(bytes, value);
                }
            }
        }
    }

    struct MetadataEntry {
        key: String,
        value: TestValue,
    }

    impl MetadataEntry {
        fn string(key: &str, value: &str) -> Self {
            Self {
                key: key.to_string(),
                value: TestValue::String(value.to_string()),
            }
        }

        fn u32(key: &str, value: u32) -> Self {
            Self {
                key: key.to_string(),
                value: TestValue::U32(value),
            }
        }
        fn array_string(key: &str, value: Vec<String>) -> Self {
            Self {
                key: key.to_string(),
                value: TestValue::ArrayString(value),
            }
        }
    }

    enum TestValue {
        U32(u32),
        String(String),
        ArrayString(Vec<String>),
    }

    struct TensorEntry {
        name: String,
        dimensions: Vec<u64>,
        ggml_type: u32,
        data: Vec<u8>,
    }
}
