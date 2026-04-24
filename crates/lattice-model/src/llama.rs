//! LLaMA-family GGUF binding helpers.

use lattice_core::{LatticeError, Result};

use crate::gguf::{GgufFile, GgufMetadata, GgufMetadataValue, GgufTensorView};

const DEFAULT_ROPE_FREQ_BASE: f64 = 10_000.0;
const DEFAULT_RMS_NORM_EPSILON: f32 = 1.0e-5;

/// A validated set of LLaMA hyperparameters required by the runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaModelSpec {
    /// Human-readable model name from `general.name`.
    pub model_name: Option<String>,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length.
    pub context_length: usize,
    /// Embedding width.
    pub embedding_length: usize,
    /// Transformer block count.
    pub block_count: usize,
    /// MLP hidden width.
    pub feed_forward_length: usize,
    /// Query attention head count.
    pub attention_head_count: usize,
    /// Key/value attention head count.
    pub attention_head_count_kv: usize,
    /// Number of rotary dimensions per head.
    pub rope_dimension_count: usize,
    /// RoPE frequency base.
    pub rope_freq_base: f64,
    /// RMSNorm epsilon used by attention and output norms.
    pub rms_norm_epsilon: f32,
    /// Query head dimension derived from embedding width and head count.
    pub head_dimension: usize,
    /// Key projection dimension per KV head.
    pub attention_key_length: usize,
    /// Value projection dimension per KV head.
    pub attention_value_length: usize,
}

impl TryFrom<&GgufMetadata> for LlamaModelSpec {
    type Error = LatticeError;

    fn try_from(metadata: &GgufMetadata) -> Result<Self> {
        let architecture = metadata.architecture_name().ok_or_else(|| {
            LatticeError::Message("missing GGUF general.architecture".to_string())
        })?;
        if architecture != "llama" {
            return Err(LatticeError::Message(format!(
                "expected llama architecture, found `{architecture}`"
            )));
        }

        let model_name = metadata
            .metadata_value("general.name")
            .and_then(GgufMetadataValue::as_str)
            .map(ToOwned::to_owned);
        let vocab_size = metadata
            .metadata_value("tokenizer.ggml.tokens")
            .and_then(GgufMetadataValue::as_array)
            .map(|tokens| tokens.len())
            .ok_or_else(|| {
                LatticeError::Message(
                    "missing GGUF tokenizer.ggml.tokens required for llama vocab size".to_string(),
                )
            })?;

        let context_length = required_usize(metadata, "llama.context_length")?;
        let embedding_length = required_usize(metadata, "llama.embedding_length")?;
        let block_count = required_usize(metadata, "llama.block_count")?;
        let feed_forward_length = required_usize(metadata, "llama.feed_forward_length")?;
        let attention_head_count = required_usize(metadata, "llama.attention.head_count")?;
        if attention_head_count == 0 {
            return Err(LatticeError::Message(
                "llama.attention.head_count must be greater than zero".to_string(),
            ));
        }
        let attention_head_count_kv = optional_usize(metadata, "llama.attention.head_count_kv")
            .unwrap_or(attention_head_count);
        if attention_head_count_kv == 0 {
            return Err(LatticeError::Message(
                "llama.attention.head_count_kv must be greater than zero".to_string(),
            ));
        }
        if !attention_head_count.is_multiple_of(attention_head_count_kv) {
            return Err(LatticeError::Message(format!(
                "llama.attention.head_count ({attention_head_count}) must be divisible by llama.attention.head_count_kv ({attention_head_count_kv})"
            )));
        }
        if !embedding_length.is_multiple_of(attention_head_count) {
            return Err(LatticeError::Message(format!(
                "llama.embedding_length ({embedding_length}) must be divisible by llama.attention.head_count ({attention_head_count})"
            )));
        }

        let head_dimension = embedding_length / attention_head_count;
        let rope_dimension_count =
            optional_usize(metadata, "llama.rope.dimension_count").unwrap_or(head_dimension);
        if rope_dimension_count > head_dimension {
            return Err(LatticeError::Message(format!(
                "llama.rope.dimension_count ({rope_dimension_count}) cannot exceed head dimension ({head_dimension})"
            )));
        }

        let attention_key_length =
            optional_usize(metadata, "llama.attention.key_length").unwrap_or(head_dimension);
        let attention_value_length =
            optional_usize(metadata, "llama.attention.value_length").unwrap_or(head_dimension);
        let rope_freq_base =
            optional_f64(metadata, "llama.rope.freq_base").unwrap_or(DEFAULT_ROPE_FREQ_BASE);
        let rms_norm_epsilon = optional_f32(metadata, "llama.attention.layer_norm_rms_epsilon")
            .or_else(|| optional_f32(metadata, "llama.attention.layer_norm_epsilon"))
            .unwrap_or(DEFAULT_RMS_NORM_EPSILON);

        Ok(Self {
            model_name,
            vocab_size,
            context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            rope_dimension_count,
            rope_freq_base,
            rms_norm_epsilon,
            head_dimension,
            attention_key_length,
            attention_value_length,
        })
    }
}

/// All required tensors for one LLaMA transformer block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaBlockTensors<'a> {
    /// Input attention RMSNorm weights.
    pub attention_norm: GgufTensorView<'a>,
    /// Attention query projection weights.
    pub attention_q: GgufTensorView<'a>,
    /// Attention key projection weights.
    pub attention_k: GgufTensorView<'a>,
    /// Attention value projection weights.
    pub attention_v: GgufTensorView<'a>,
    /// Attention output projection weights.
    pub attention_output: GgufTensorView<'a>,
    /// Feed-forward RMSNorm weights.
    pub ffn_norm: GgufTensorView<'a>,
    /// Feed-forward gate projection weights.
    pub ffn_gate: GgufTensorView<'a>,
    /// Feed-forward down projection weights.
    pub ffn_down: GgufTensorView<'a>,
    /// Feed-forward up projection weights.
    pub ffn_up: GgufTensorView<'a>,
}

/// Bound tensor views required to execute a LLaMA-family GGUF model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaTensorMap<'a> {
    /// Token embedding matrix.
    pub token_embeddings: GgufTensorView<'a>,
    /// Final output RMSNorm weights.
    pub output_norm: GgufTensorView<'a>,
    /// Final output projection weights.
    pub output: GgufTensorView<'a>,
    /// Whether the output projection reuses the token embedding weights.
    pub output_is_tied: bool,
    /// Per-layer transformer block weights.
    pub blocks: Vec<LlamaBlockTensors<'a>>,
}

impl<'a> LlamaTensorMap<'a> {
    /// Binds all required LLaMA tensor views from a parsed GGUF file.
    pub fn from_gguf(file: &GgufFile<'a>, spec: &LlamaModelSpec) -> Result<Self> {
        let token_embeddings = require_tensor(file, "token_embd.weight")?;
        let output_norm = require_tensor(file, "output_norm.weight")?;
        let (output, output_is_tied) = resolve_output_tensor(file, &token_embeddings)?;

        require_vector_len(&output_norm, spec.embedding_length, "output_norm.weight")?;
        require_matrix_inner_dim(
            &token_embeddings,
            spec.embedding_length,
            "token_embd.weight",
        )?;
        require_matrix_inner_dim(&output, spec.embedding_length, "output.weight")?;

        let mut blocks = Vec::with_capacity(spec.block_count);
        for block_index in 0..spec.block_count {
            let attention_norm = require_block_tensor(file, block_index, "attn_norm.weight")?;
            let attention_q = require_block_tensor(file, block_index, "attn_q.weight")?;
            let attention_k = require_block_tensor(file, block_index, "attn_k.weight")?;
            let attention_v = require_block_tensor(file, block_index, "attn_v.weight")?;
            let attention_output = require_block_tensor(file, block_index, "attn_output.weight")?;
            let ffn_norm = require_block_tensor(file, block_index, "ffn_norm.weight")?;
            let ffn_gate = require_block_tensor(file, block_index, "ffn_gate.weight")?;
            let ffn_down = require_block_tensor(file, block_index, "ffn_down.weight")?;
            let ffn_up = require_block_tensor(file, block_index, "ffn_up.weight")?;

            require_vector_len(
                &attention_norm,
                spec.embedding_length,
                &format!("blk.{block_index}.attn_norm.weight"),
            )?;
            require_vector_len(
                &ffn_norm,
                spec.embedding_length,
                &format!("blk.{block_index}.ffn_norm.weight"),
            )?;

            blocks.push(LlamaBlockTensors {
                attention_norm,
                attention_q,
                attention_k,
                attention_v,
                attention_output,
                ffn_norm,
                ffn_gate,
                ffn_down,
                ffn_up,
            });
        }

        Ok(Self {
            token_embeddings,
            output_norm,
            output,
            output_is_tied,
            blocks,
        })
    }

    /// Returns a block by index.
    pub fn block(&self, index: usize) -> Option<&LlamaBlockTensors<'a>> {
        self.blocks.get(index)
    }
}

/// A validated, runtime-ready LLaMA binding over a GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaModel<'a> {
    /// Validated hyperparameters.
    pub spec: LlamaModelSpec,
    /// Bound tensor views.
    pub tensors: LlamaTensorMap<'a>,
}

impl<'a> LlamaModel<'a> {
    /// Builds a runtime-ready LLaMA model binding from a parsed GGUF file.
    pub fn from_gguf(file: &GgufFile<'a>) -> Result<Self> {
        let spec = LlamaModelSpec::try_from(file.metadata())?;
        let tensors = LlamaTensorMap::from_gguf(file, &spec)?;
        Ok(Self { spec, tensors })
    }
}

fn require_tensor<'a>(file: &GgufFile<'a>, name: &str) -> Result<GgufTensorView<'a>> {
    file.tensor(name)?
        .ok_or_else(|| LatticeError::Message(format!("missing required llama tensor `{name}`")))
}

fn resolve_output_tensor<'a>(
    file: &GgufFile<'a>,
    token_embeddings: &GgufTensorView<'a>,
) -> Result<(GgufTensorView<'a>, bool)> {
    if let Some(output) = optional_tensor(file, &["output.weight", "output"])? {
        return Ok((output, false));
    }

    Ok((token_embeddings.clone(), true))
}

fn require_block_tensor<'a>(
    file: &GgufFile<'a>,
    block_index: usize,
    suffix: &str,
) -> Result<GgufTensorView<'a>> {
    let name = format!("blk.{block_index}.{suffix}");
    require_tensor(file, &name)
}

fn require_vector_len(
    tensor: &GgufTensorView<'_>,
    expected: usize,
    logical_name: &str,
) -> Result<()> {
    let dimensions = tensor.dimensions();
    if dimensions.len() != 1 || dimensions[0] != expected as u64 {
        return Err(LatticeError::Message(format!(
            "llama tensor `{logical_name}` expected shape [{expected}], found {:?}",
            dimensions
        )));
    }
    Ok(())
}

fn require_matrix_inner_dim(
    tensor: &GgufTensorView<'_>,
    expected: usize,
    logical_name: &str,
) -> Result<()> {
    let Some(inner_dim) = tensor.dimensions().first() else {
        return Err(LatticeError::Message(format!(
            "llama tensor `{logical_name}` must have at least one dimension"
        )));
    };
    if *inner_dim != expected as u64 {
        return Err(LatticeError::Message(format!(
            "llama tensor `{logical_name}` expected inner dimension {expected}, found {}",
            inner_dim
        )));
    }
    Ok(())
}

fn required_usize(metadata: &GgufMetadata, key: &str) -> Result<usize> {
    optional_usize(metadata, key).ok_or_else(|| {
        LatticeError::Message(format!("missing required llama metadata key `{key}`"))
    })
}

fn optional_tensor<'a>(
    file: &GgufFile<'a>,
    names: &[&str],
) -> Result<Option<GgufTensorView<'a>>> {
    for name in names {
        if let Some(tensor) = file.tensor(name)? {
            return Ok(Some(tensor));
        }
    }

    Ok(None)
}

fn optional_usize(metadata: &GgufMetadata, key: &str) -> Option<usize> {
    metadata
        .metadata_value(key)
        .and_then(GgufMetadataValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn optional_f32(metadata: &GgufMetadata, key: &str) -> Option<f32> {
    match metadata.metadata_value(key) {
        Some(GgufMetadataValue::F32(value)) => Some(*value),
        Some(GgufMetadataValue::F64(value)) => Some(*value as f32),
        _ => None,
    }
}

fn optional_f64(metadata: &GgufMetadata, key: &str) -> Option<f64> {
    metadata
        .metadata_value(key)
        .and_then(GgufMetadataValue::as_f64)
}

#[cfg(test)]
mod tests {
    use crate::gguf::{GgmlType, GgufMetadataValueType, parse_gguf};

    use super::{LlamaModel, LlamaModelSpec};

    #[test]
    fn builds_validated_llama_spec_with_defaults() {
        let bytes = build_llama_gguf(1, None, true, false);
        let gguf = parse_gguf(&bytes).expect("GGUF should parse");

        let spec = LlamaModelSpec::try_from(gguf.metadata()).expect("llama spec should parse");
        assert_eq!(spec.model_name.as_deref(), Some("TinyLlama"));
        assert_eq!(spec.vocab_size, 32);
        assert_eq!(spec.context_length, 128);
        assert_eq!(spec.embedding_length, 8);
        assert_eq!(spec.block_count, 1);
        assert_eq!(spec.feed_forward_length, 16);
        assert_eq!(spec.attention_head_count, 2);
        assert_eq!(spec.attention_head_count_kv, 2);
        assert_eq!(spec.head_dimension, 4);
        assert_eq!(spec.rope_dimension_count, 4);
        assert_eq!(spec.rope_freq_base, 10_000.0);
        assert_eq!(spec.rms_norm_epsilon, 1.0e-5);
    }

    #[test]
    fn binds_required_llama_tensors() {
        let bytes = build_llama_gguf(2, Some("output.weight"), true, true);
        let gguf = parse_gguf(&bytes).expect("GGUF should parse");

        let model = LlamaModel::from_gguf(&gguf).expect("llama model should bind");
        assert_eq!(model.spec.block_count, 2);
        assert_eq!(model.tensors.blocks.len(), 2);
        assert!(!model.tensors.output_is_tied);
        assert_eq!(model.tensors.token_embeddings.name(), "token_embd.weight");
        assert_eq!(model.tensors.output.name(), "output.weight");
        assert_eq!(
            model
                .tensors
                .block(1)
                .expect("block should exist")
                .ffn_up
                .name(),
            "blk.1.ffn_up.weight"
        );
    }

    #[test]
    fn binds_legacy_output_tensor_name() {
        let bytes = build_llama_gguf(1, Some("output"), true, true);
        let gguf = parse_gguf(&bytes).expect("GGUF should parse");

        let model = LlamaModel::from_gguf(&gguf).expect("legacy output tensor should bind");
        assert_eq!(model.tensors.output.name(), "output");
        assert!(!model.tensors.output_is_tied);
    }

    #[test]
    fn ties_output_to_token_embeddings_when_output_tensor_is_missing() {
        let bytes = build_llama_gguf(1, None, true, true);
        let gguf = parse_gguf(&bytes).expect("GGUF should parse");

        let model = LlamaModel::from_gguf(&gguf).expect("missing output tensor should tie");
        assert_eq!(model.tensors.output.name(), "token_embd.weight");
        assert!(model.tensors.output_is_tied);
    }

    #[test]
    fn rejects_missing_required_tensor() {
        let bytes = build_llama_gguf(1, Some("output.weight"), false, true);
        let gguf = parse_gguf(&bytes).expect("GGUF should parse");

        let error = LlamaModel::from_gguf(&gguf).expect_err("missing tensor should fail");
        assert!(
            error
                .to_string()
                .contains("missing required llama tensor `output_norm.weight`"),
            "unexpected error: {error}"
        );
    }

    fn build_llama_gguf(
        block_count: usize,
        output_name: Option<&str>,
        include_output_norm: bool,
        with_optional_metadata: bool,
    ) -> Vec<u8> {
        let embedding = 8_usize;
        let feed_forward = 16_usize;
        let vocab = 32_usize;

        let mut metadata = vec![
            MetadataEntry::string("general.architecture", "llama"),
            MetadataEntry::string("general.name", "TinyLlama"),
            MetadataEntry::u32("general.alignment", 32),
            MetadataEntry::u32("llama.context_length", 128),
            MetadataEntry::u32("llama.embedding_length", embedding as u32),
            MetadataEntry::u32("llama.block_count", block_count as u32),
            MetadataEntry::u32("llama.feed_forward_length", feed_forward as u32),
            MetadataEntry::u32("llama.attention.head_count", 2),
            MetadataEntry::array_string(
                "tokenizer.ggml.tokens",
                (0..vocab)
                    .map(|index| format!("tok-{index}"))
                    .collect::<Vec<_>>(),
            ),
        ];

        if with_optional_metadata {
            metadata.push(MetadataEntry::u32("llama.attention.head_count_kv", 1));
            metadata.push(MetadataEntry::u32("llama.rope.dimension_count", 4));
            metadata.push(MetadataEntry::f32("llama.rope.freq_base", 5_000.0));
            metadata.push(MetadataEntry::f32(
                "llama.attention.layer_norm_rms_epsilon",
                1.0e-6,
            ));
        }

        let mut tensors = vec![tensor(
            "token_embd.weight",
            vec![embedding as u64, vocab as u64],
        )];
        if include_output_norm {
            tensors.push(tensor("output_norm.weight", vec![embedding as u64]));
        }
        if let Some(output_name) = output_name {
            tensors.push(tensor(output_name, vec![embedding as u64, vocab as u64]));
        }

        for block in 0..block_count {
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
        let byte_len = GgmlType::F32
            .tensor_byte_len(&dimensions)
            .expect("tensor byte length should fit");
        TensorEntry {
            name: name.to_string(),
            dimensions,
            ggml_type: GgmlType::F32.raw(),
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

        while bytes.len() % alignment != 0 {
            bytes.push(0);
        }

        let mut tensor_data = vec![0_u8; next_offset.max(alignment)];
        for (tensor, offset) in tensor_entries.iter().zip(aligned_offsets.iter()) {
            let offset = *offset as usize;
            let end = offset + tensor.data.len();
            tensor_data[offset..end].copy_from_slice(&tensor.data);
        }

        bytes.extend_from_slice(&tensor_data);
        bytes
    }

    fn align_to(offset: usize, alignment: usize) -> usize {
        let mask = alignment - 1;
        (offset + mask) & !mask
    }

    fn push_value(bytes: &mut Vec<u8>, value: &TestValue) {
        match value {
            TestValue::String(value) => {
                push_u32(bytes, GgufMetadataValueType::String as u32);
                push_string(bytes, value);
            }
            TestValue::U32(value) => {
                push_u32(bytes, GgufMetadataValueType::U32 as u32);
                push_u32(bytes, *value);
            }
            TestValue::F32(value) => {
                push_u32(bytes, GgufMetadataValueType::F32 as u32);
                push_f32(bytes, *value);
            }
            TestValue::StringArray(values) => {
                push_u32(bytes, GgufMetadataValueType::Array as u32);
                push_u32(bytes, GgufMetadataValueType::String as u32);
                push_u64(bytes, values.len() as u64);
                for value in values {
                    push_string(bytes, value);
                }
            }
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

    fn push_f32(bytes: &mut Vec<u8>, value: f32) {
        push_u32(bytes, value.to_bits());
    }

    struct TensorEntry {
        name: String,
        dimensions: Vec<u64>,
        ggml_type: u32,
        data: Vec<u8>,
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

        fn f32(key: &str, value: f32) -> Self {
            Self {
                key: key.to_string(),
                value: TestValue::F32(value),
            }
        }

        fn array_string(key: &str, values: Vec<String>) -> Self {
            Self {
                key: key.to_string(),
                value: TestValue::StringArray(values),
            }
        }
    }

    enum TestValue {
        String(String),
        U32(u32),
        F32(f32),
        StringArray(Vec<String>),
    }
}
