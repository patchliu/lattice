use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use lattice_model::MappedWeights;

use super::prefill;

#[test]
fn prefills_identity_tied_model() {
    let path = write_temp_file("llama-prefill", &build_identity_llama_gguf());
    let weights = MappedWeights::open(&path).expect("weights should map");
    let gguf = weights.gguf_file().expect("GGUF should parse");
    let model = lattice_model::LlamaModel::from_gguf(&gguf).expect("llama should bind");

    let output = prefill(&model, &[0, 1, 2]).expect("prefill should work");
    assert_eq!(output.next_token_id, 2);
    assert_eq!(output.logits.len(), 3);
    assert!(output.logits[2] > output.logits[1]);
    assert!(output.logits[2] > output.logits[0]);

    fs::remove_file(path).expect("temp file should be removed");
}

fn write_temp_file(prefix: &str, bytes: &[u8]) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should move forward")
        .as_nanos();
    path.push(format!("{prefix}-{unique}.gguf"));
    fs::write(&path, bytes).expect("temp GGUF should be written");
    path
}

fn build_identity_llama_gguf() -> Vec<u8> {
    let embedding = 4_u64;
    let feed_forward = 4_u64;
    let vocab = 3_u64;

    let metadata = vec![
        MetadataEntry::string("general.architecture", "llama"),
        MetadataEntry::string("general.name", "IdentityLlama"),
        MetadataEntry::u32("general.alignment", 32),
        MetadataEntry::u32("llama.context_length", 16),
        MetadataEntry::u32("llama.embedding_length", embedding as u32),
        MetadataEntry::u32("llama.block_count", 1),
        MetadataEntry::u32("llama.feed_forward_length", feed_forward as u32),
        MetadataEntry::u32("llama.attention.head_count", 1),
        MetadataEntry::u32("llama.attention.head_count_kv", 1),
        MetadataEntry::u32("llama.rope.dimension_count", 2),
        MetadataEntry::f32("llama.rope.freq_base", 10_000.0),
        MetadataEntry::f32("llama.attention.layer_norm_rms_epsilon", 1.0e-5),
        MetadataEntry::array_string(
            "tokenizer.ggml.tokens",
            vec!["tok-0".to_string(), "tok-1".to_string(), "tok-2".to_string()],
        ),
    ];

    let tensors = vec![
        tensor_f32(
            "token_embd.weight",
            vec![embedding, vocab],
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.5, 0.0,
            ],
        ),
        tensor_f32("output_norm.weight", vec![embedding], &[1.0, 1.0, 1.0, 1.0]),
        tensor_f32("blk.0.attn_norm.weight", vec![embedding], &[1.0, 1.0, 1.0, 1.0]),
        tensor_f32("blk.0.attn_q.weight", vec![embedding, embedding], &[0.0; 16]),
        tensor_f32("blk.0.attn_k.weight", vec![embedding, embedding], &[0.0; 16]),
        tensor_f32("blk.0.attn_v.weight", vec![embedding, embedding], &[0.0; 16]),
        tensor_f32("blk.0.attn_output.weight", vec![embedding, embedding], &[0.0; 16]),
        tensor_f32("blk.0.ffn_norm.weight", vec![embedding], &[1.0, 1.0, 1.0, 1.0]),
        tensor_f32("blk.0.ffn_gate.weight", vec![embedding, feed_forward], &[0.0; 16]),
        tensor_f32("blk.0.ffn_down.weight", vec![feed_forward, embedding], &[0.0; 16]),
        tensor_f32("blk.0.ffn_up.weight", vec![embedding, feed_forward], &[0.0; 16]),
    ];

    build_gguf(&metadata, &tensors, 32)
}

fn tensor_f32(name: &str, dimensions: Vec<u64>, values: &[f32]) -> TensorEntry {
    let mut data = Vec::with_capacity(values.len() * 4);
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }

    TensorEntry {
        name: name.to_string(),
        dimensions,
        ggml_type: lattice_model::GgmlType::F32.raw(),
        data,
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

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
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
        TestValue::F32(value) => {
            push_u32(bytes, 6);
            push_f32(bytes, *value);
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

    fn f32(key: &str, value: f32) -> Self {
        Self {
            key: key.to_string(),
            value: TestValue::F32(value),
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
    F32(f32),
    String(String),
    ArrayString(Vec<String>),
}

struct TensorEntry {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}
