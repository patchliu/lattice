//! GGUF metadata parsing primitives.
#![allow(non_camel_case_types)]

use std::collections::BTreeMap;
use std::ops::Range;

use lattice_core::{LatticeError, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: usize = 32;

/// Parsed GGUF file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

impl GgufHeader {
    /// Returns the GGUF file version.
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the number of tensor descriptors stored in the file.
    pub const fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    /// Returns the number of metadata key/value pairs.
    pub const fn metadata_kv_count(&self) -> u64 {
        self.metadata_kv_count
    }
}

/// The scalar type used to encode a metadata value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufMetadataValueType {
    /// Unsigned 8-bit integer.
    U8 = 0,
    /// Signed 8-bit integer.
    I8 = 1,
    /// Unsigned 16-bit integer.
    U16 = 2,
    /// Signed 16-bit integer.
    I16 = 3,
    /// Unsigned 32-bit integer.
    U32 = 4,
    /// Signed 32-bit integer.
    I32 = 5,
    /// 32-bit float.
    F32 = 6,
    /// Boolean.
    Bool = 7,
    /// UTF-8 string.
    String = 8,
    /// Homogeneous array.
    Array = 9,
    /// Unsigned 64-bit integer.
    U64 = 10,
    /// Signed 64-bit integer.
    I64 = 11,
    /// 64-bit float.
    F64 = 12,
}

impl GgufMetadataValueType {
    fn from_raw(raw: u32) -> Result<Self> {
        let value_type = match raw {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => {
                return Err(LatticeError::Message(format!(
                    "unsupported GGUF metadata value type: {raw}"
                )));
            }
        };

        Ok(value_type)
    }
}

/// A parsed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Homogeneous array.
    Array(Vec<GgufMetadataValue>),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit float.
    F64(f64),
}

impl GgufMetadataValue {
    /// Returns the raw GGUF type tag for the value.
    pub const fn value_type(&self) -> GgufMetadataValueType {
        match self {
            Self::U8(_) => GgufMetadataValueType::U8,
            Self::I8(_) => GgufMetadataValueType::I8,
            Self::U16(_) => GgufMetadataValueType::U16,
            Self::I16(_) => GgufMetadataValueType::I16,
            Self::U32(_) => GgufMetadataValueType::U32,
            Self::I32(_) => GgufMetadataValueType::I32,
            Self::F32(_) => GgufMetadataValueType::F32,
            Self::Bool(_) => GgufMetadataValueType::Bool,
            Self::String(_) => GgufMetadataValueType::String,
            Self::Array(_) => GgufMetadataValueType::Array,
            Self::U64(_) => GgufMetadataValueType::U64,
            Self::I64(_) => GgufMetadataValueType::I64,
            Self::F64(_) => GgufMetadataValueType::F64,
        }
    }

    /// Returns the value as a string slice if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Returns the value as a `u64` when it is represented as a non-negative integer.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(u64::from(*value)),
            Self::U16(value) => Some(u64::from(*value)),
            Self::U32(value) => Some(u64::from(*value)),
            Self::U64(value) => Some(*value),
            Self::I8(value) if *value >= 0 => Some(*value as u64),
            Self::I16(value) if *value >= 0 => Some(*value as u64),
            Self::I32(value) if *value >= 0 => Some(*value as u64),
            Self::I64(value) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }

    /// Returns the value as a `f64` if it is a float.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F32(value) => Some(f64::from(*value)),
            Self::F64(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the value as an array if it is an array.
    pub fn as_array(&self) -> Option<&[GgufMetadataValue]> {
        match self {
            Self::Array(values) => Some(values.as_slice()),
            _ => None,
        }
    }
}

/// Per-tensor metadata stored in the GGUF directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    offset: u64,
}

impl GgufTensorInfo {
    /// Returns the tensor name.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns the logical tensor dimensions.
    pub fn dimensions(&self) -> &[u64] {
        self.dimensions.as_slice()
    }

    /// Returns the raw GGML tensor type id.
    pub const fn ggml_type(&self) -> u32 {
        self.ggml_type
    }

    /// Returns the parsed GGML tensor type if this type is currently supported.
    pub fn parsed_ggml_type(&self) -> Result<GgmlType> {
        GgmlType::from_raw(self.ggml_type)
    }

    /// Returns the tensor data offset relative to the GGUF tensor data section.
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the total number of logical tensor elements.
    pub fn element_count(&self) -> Result<u64> {
        self.dimensions.iter().try_fold(1_u64, |acc, dimension| {
            acc.checked_mul(*dimension).ok_or_else(|| {
                LatticeError::Message(format!("tensor `{}` element count overflowed", self.name))
            })
        })
    }

    /// Returns the encoded tensor byte length based on the GGML type and shape.
    pub fn byte_len(&self) -> Result<usize> {
        self.parsed_ggml_type()?.tensor_byte_len(&self.dimensions)
    }

    /// Returns the absolute byte offset from the start of the file.
    pub fn file_offset(&self, tensor_data_offset: usize) -> Result<usize> {
        tensor_data_offset
            .checked_add(usize::try_from(self.offset).map_err(|_| {
                LatticeError::Message(format!(
                    "tensor `{}` offset does not fit into usize",
                    self.name
                ))
            })?)
            .ok_or_else(|| {
                LatticeError::Message(format!("tensor `{}` absolute offset overflowed", self.name))
            })
    }
}

/// A GGML tensor encoding supported by the GGUF parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// Q4_0 quantization.
    Q4_0,
    /// Q4_1 quantization.
    Q4_1,
    /// Q5_0 quantization.
    Q5_0,
    /// Q5_1 quantization.
    Q5_1,
    /// Q8_0 quantization.
    Q8_0,
    /// Q8_1 quantization.
    Q8_1,
    /// Q2_K quantization.
    Q2_K,
    /// Q3_K quantization.
    Q3_K,
    /// Q4_K quantization.
    Q4_K,
    /// Q5_K quantization.
    Q5_K,
    /// Q6_K quantization.
    Q6_K,
}

impl GgmlType {
    /// Parses a GGML type id from GGUF tensor metadata.
    pub fn from_raw(raw: u32) -> Result<Self> {
        let value = match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            _ => {
                return Err(LatticeError::Message(format!(
                    "unsupported GGML tensor type id: {raw}"
                )));
            }
        };

        Ok(value)
    }

    /// Returns the raw GGML type id used in the GGUF file.
    pub const fn raw(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
        }
    }

    /// Returns the GGML storage layout for the tensor type.
    pub const fn layout(self) -> GgmlTypeLayout {
        match self {
            Self::F32 => GgmlTypeLayout::new(1, 4),
            Self::F16 => GgmlTypeLayout::new(1, 2),
            Self::Q4_0 => GgmlTypeLayout::new(32, 18),
            Self::Q4_1 => GgmlTypeLayout::new(32, 20),
            Self::Q5_0 => GgmlTypeLayout::new(32, 22),
            Self::Q5_1 => GgmlTypeLayout::new(32, 24),
            Self::Q8_0 => GgmlTypeLayout::new(32, 34),
            Self::Q8_1 => GgmlTypeLayout::new(32, 36),
            Self::Q2_K => GgmlTypeLayout::new(256, 84),
            Self::Q3_K => GgmlTypeLayout::new(256, 110),
            Self::Q4_K => GgmlTypeLayout::new(256, 144),
            Self::Q5_K => GgmlTypeLayout::new(256, 176),
            Self::Q6_K => GgmlTypeLayout::new(256, 210),
        }
    }

    /// Returns whether the type is quantized.
    pub const fn is_quantized(self) -> bool {
        self.layout().block_size > 1
    }

    /// Computes the encoded byte length for a tensor with the provided shape.
    pub fn tensor_byte_len(self, dimensions: &[u64]) -> Result<usize> {
        let layout = self.layout();
        let row_elements = dimensions.first().copied().unwrap_or(1);
        let row_elements = usize::try_from(row_elements).map_err(|_| {
            LatticeError::Message("GGUF tensor row width does not fit into usize".to_string())
        })?;
        if row_elements == 0 {
            return Ok(0);
        }
        if row_elements % layout.block_size != 0 {
            return Err(LatticeError::Message(format!(
                "tensor row width {row_elements} is not divisible by block size {} for {:?}",
                layout.block_size, self
            )));
        }

        let row_bytes = row_elements
            .checked_div(layout.block_size)
            .and_then(|blocks| blocks.checked_mul(layout.bytes_per_block))
            .ok_or_else(|| {
                LatticeError::Message("GGUF tensor row byte length overflowed".to_string())
            })?;

        let row_count = dimensions
            .iter()
            .skip(1)
            .try_fold(1_usize, |acc, dimension| {
                let dimension = usize::try_from(*dimension).map_err(|_| {
                    LatticeError::Message(
                        "GGUF tensor dimension does not fit into usize".to_string(),
                    )
                })?;
                acc.checked_mul(dimension).ok_or_else(|| {
                    LatticeError::Message("GGUF tensor row count overflowed".to_string())
                })
            })?;

        row_bytes
            .checked_mul(row_count)
            .ok_or_else(|| LatticeError::Message("GGUF tensor byte length overflowed".to_string()))
    }
}

/// The encoded storage layout of a GGML tensor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlTypeLayout {
    block_size: usize,
    bytes_per_block: usize,
}

impl GgmlTypeLayout {
    const fn new(block_size: usize, bytes_per_block: usize) -> Self {
        Self {
            block_size,
            bytes_per_block,
        }
    }

    /// Returns the number of logical values encoded in one storage block.
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the encoded byte width of one storage block.
    pub const fn bytes_per_block(&self) -> usize {
        self.bytes_per_block
    }
}

/// A parsed GGUF file view containing generic metadata and tensor directory entries.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufMetadata {
    header: GgufHeader,
    metadata: BTreeMap<String, GgufMetadataValue>,
    tensor_infos: Vec<GgufTensorInfo>,
    tensor_indices: BTreeMap<String, usize>,
    tensor_data_offset: usize,
}

impl GgufMetadata {
    /// Parses GGUF metadata from a memory buffer.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let magic = reader.read_exact(4)?;
        if magic != GGUF_MAGIC {
            return Err(LatticeError::Message(format!(
                "invalid GGUF magic: expected {:?}, got {:?}",
                GGUF_MAGIC, magic
            )));
        }

        let header = GgufHeader {
            version: reader.read_u32()?,
            tensor_count: reader.read_u64()?,
            metadata_kv_count: reader.read_u64()?,
        };

        let mut metadata = BTreeMap::new();
        for _ in 0..header.metadata_kv_count {
            let key = reader.read_string()?;
            if metadata.contains_key(&key) {
                return Err(LatticeError::Message(format!(
                    "duplicate GGUF metadata key `{key}`"
                )));
            }
            let value_type = GgufMetadataValueType::from_raw(reader.read_u32()?)?;
            let value = reader.read_value(value_type)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = Vec::with_capacity(header.tensor_count as usize);
        let mut tensor_indices = BTreeMap::new();
        for _ in 0..header.tensor_count {
            let name = reader.read_string()?;
            let dimension_count = usize::try_from(reader.read_u32()?).map_err(|_| {
                LatticeError::Message(format!(
                    "tensor `{name}` dimension count does not fit into usize"
                ))
            })?;
            let mut dimensions = Vec::with_capacity(dimension_count);
            for _ in 0..dimension_count {
                dimensions.push(reader.read_u64()?);
            }

            if tensor_indices.contains_key(&name) {
                return Err(LatticeError::Message(format!(
                    "duplicate GGUF tensor name `{name}`"
                )));
            }

            tensor_infos.push(GgufTensorInfo {
                name,
                dimensions,
                ggml_type: reader.read_u32()?,
                offset: reader.read_u64()?,
            });
            let index = tensor_infos.len() - 1;
            tensor_indices.insert(tensor_infos[index].name.clone(), index);
        }

        let tensor_data_offset = align_offset(reader.offset(), metadata_alignment(&metadata)?)?;
        Ok(Self {
            header,
            metadata,
            tensor_infos,
            tensor_indices,
            tensor_data_offset,
        })
    }

    /// Returns the parsed header.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Returns all metadata entries.
    pub fn metadata(&self) -> &BTreeMap<String, GgufMetadataValue> {
        &self.metadata
    }

    /// Returns a metadata value by key.
    pub fn metadata_value(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata.get(key)
    }

    /// Returns the model architecture name from `general.architecture` if present.
    pub fn architecture_name(&self) -> Option<&str> {
        self.metadata_value("general.architecture")
            .and_then(GgufMetadataValue::as_str)
    }

    /// Returns a typed architecture view for the parsed metadata.
    pub fn architecture_metadata(&self) -> Option<ArchitectureMetadata> {
        match self.architecture_name()? {
            "llama" => Some(ArchitectureMetadata::Llama(LlamaMetadata::from_gguf(self))),
            architecture => Some(ArchitectureMetadata::Unknown(architecture.to_owned())),
        }
    }

    /// Returns the tensor alignment recorded in metadata or the GGUF default.
    pub fn alignment(&self) -> Result<usize> {
        metadata_alignment(&self.metadata)
    }

    /// Returns the parsed tensor directory.
    pub fn tensor_infos(&self) -> &[GgufTensorInfo] {
        self.tensor_infos.as_slice()
    }

    /// Returns a tensor descriptor by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_indices
            .get(name)
            .and_then(|index| self.tensor_infos.get(*index))
    }

    /// Returns the absolute file offset of the tensor data section.
    pub const fn tensor_data_offset(&self) -> usize {
        self.tensor_data_offset
    }
}

/// A parsed GGUF file that can expose zero-copy tensor views.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile<'a> {
    metadata: GgufMetadata,
    bytes: &'a [u8],
}

impl<'a> GgufFile<'a> {
    /// Parses a GGUF file from raw bytes.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        Ok(Self {
            metadata: GgufMetadata::parse(bytes)?,
            bytes,
        })
    }

    /// Returns the parsed metadata.
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Returns a zero-copy view over a named tensor if present.
    pub fn tensor(&'a self, name: &str) -> Result<Option<GgufTensorView<'a>>> {
        let Some(info) = self.metadata.tensor_info(name) else {
            return Ok(None);
        };

        Ok(Some(self.tensor_view(info)?))
    }

    fn tensor_view(&'a self, info: &'a GgufTensorInfo) -> Result<GgufTensorView<'a>> {
        let alignment = self.metadata.alignment()?;
        let offset = usize::try_from(info.offset).map_err(|_| {
            LatticeError::Message(format!(
                "tensor `{}` offset does not fit into usize",
                info.name()
            ))
        })?;
        if offset % alignment != 0 {
            return Err(LatticeError::Message(format!(
                "tensor `{}` offset {} is not aligned to {}",
                info.name(),
                offset,
                alignment
            )));
        }

        let byte_len = info.byte_len()?;
        let start = info.file_offset(self.metadata.tensor_data_offset())?;
        let end = start.checked_add(byte_len).ok_or_else(|| {
            LatticeError::Message(format!(
                "tensor `{}` range overflowed while building view",
                info.name()
            ))
        })?;
        let data = self.bytes.get(start..end).ok_or_else(|| {
            LatticeError::Message(format!(
                "tensor `{}` range {}..{} exceeds GGUF file size {}",
                info.name(),
                start,
                end,
                self.bytes.len()
            ))
        })?;

        Ok(GgufTensorView {
            info,
            data,
            file_range: start..end,
        })
    }
}

/// A zero-copy view over an encoded GGUF tensor payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorView<'a> {
    info: &'a GgufTensorInfo,
    data: &'a [u8],
    file_range: Range<usize>,
}

impl<'a> GgufTensorView<'a> {
    /// Returns the tensor descriptor.
    pub fn info(&self) -> &GgufTensorInfo {
        self.info
    }

    /// Returns the tensor name.
    pub fn name(&self) -> &str {
        self.info.name()
    }

    /// Returns the logical tensor dimensions.
    pub fn dimensions(&self) -> &[u64] {
        self.info.dimensions()
    }

    /// Returns the raw GGML type id.
    pub const fn ggml_type_raw(&self) -> u32 {
        self.info.ggml_type()
    }

    /// Returns the parsed GGML type.
    pub fn ggml_type(&self) -> Result<GgmlType> {
        self.info.parsed_ggml_type()
    }

    /// Returns the encoded tensor bytes.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Returns the tensor byte range in the underlying file.
    pub fn file_range(&self) -> Range<usize> {
        self.file_range.clone()
    }

    /// Returns the encoded tensor byte length.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

/// An architecture-specific metadata view extracted from the generic GGUF metadata map.
#[derive(Debug, Clone, PartialEq)]
pub enum ArchitectureMetadata {
    /// LLaMA-compatible architecture metadata.
    Llama(LlamaMetadata),
    /// An architecture string that does not have a typed parser yet.
    Unknown(String),
}

/// A typed view over commonly used LLaMA-family GGUF metadata keys.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlamaMetadata {
    /// Human-readable model name from `general.name`.
    pub model_name: Option<String>,
    /// Maximum context length.
    pub context_length: Option<u64>,
    /// Embedding width.
    pub embedding_length: Option<u64>,
    /// Transformer block count.
    pub block_count: Option<u64>,
    /// MLP hidden width.
    pub feed_forward_length: Option<u64>,
    /// Attention head count.
    pub attention_head_count: Option<u64>,
    /// Grouped-query attention KV head count.
    pub attention_head_count_kv: Option<u64>,
    /// Optional RoPE dimension count.
    pub rope_dimension_count: Option<u64>,
    /// Optional RoPE frequency base.
    pub rope_freq_base: Option<f64>,
    /// Optional vocabulary size.
    pub vocab_size: Option<u64>,
}

impl LlamaMetadata {
    /// Extracts a LLaMA-family metadata view from generic GGUF metadata.
    pub fn from_gguf(metadata: &GgufMetadata) -> Self {
        Self {
            model_name: metadata
                .metadata_value("general.name")
                .and_then(GgufMetadataValue::as_str)
                .map(ToOwned::to_owned),
            context_length: metadata_u64(metadata, "llama.context_length"),
            embedding_length: metadata_u64(metadata, "llama.embedding_length"),
            block_count: metadata_u64(metadata, "llama.block_count"),
            feed_forward_length: metadata_u64(metadata, "llama.feed_forward_length"),
            attention_head_count: metadata_u64(metadata, "llama.attention.head_count"),
            attention_head_count_kv: metadata_u64(metadata, "llama.attention.head_count_kv"),
            rope_dimension_count: metadata_u64(metadata, "llama.rope.dimension_count"),
            rope_freq_base: metadata_f64(metadata, "llama.rope.freq_base"),
            vocab_size: metadata
                .metadata_value("tokenizer.ggml.tokens")
                .and_then(GgufMetadataValue::as_array)
                .map(|tokens| tokens.len() as u64),
        }
    }
}

/// Parses GGUF metadata from a byte slice.
pub fn parse_gguf_metadata(bytes: &[u8]) -> Result<GgufMetadata> {
    GgufMetadata::parse(bytes)
}

/// Parses a GGUF file and enables zero-copy tensor lookup.
pub fn parse_gguf(bytes: &[u8]) -> Result<GgufFile<'_>> {
    GgufFile::parse(bytes)
}

fn metadata_u64(metadata: &GgufMetadata, key: &str) -> Option<u64> {
    metadata
        .metadata_value(key)
        .and_then(GgufMetadataValue::as_u64)
}

fn metadata_f64(metadata: &GgufMetadata, key: &str) -> Option<f64> {
    metadata
        .metadata_value(key)
        .and_then(GgufMetadataValue::as_f64)
}

fn metadata_alignment(metadata: &BTreeMap<String, GgufMetadataValue>) -> Result<usize> {
    let Some(raw_alignment) = metadata
        .get("general.alignment")
        .and_then(GgufMetadataValue::as_u64)
    else {
        return Ok(DEFAULT_ALIGNMENT);
    };

    let alignment = usize::try_from(raw_alignment)
        .map_err(|_| LatticeError::Message("GGUF alignment does not fit into usize".to_string()))?;
    if alignment == 0 {
        return Err(LatticeError::Message(
            "GGUF alignment must be greater than zero".to_string(),
        ));
    }
    if !alignment.is_power_of_two() {
        return Err(LatticeError::Message(format!(
            "GGUF alignment must be a power of two, got {alignment}"
        )));
    }

    Ok(alignment)
}

fn align_offset(offset: usize, alignment: usize) -> Result<usize> {
    let mask = alignment - 1;
    offset
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| LatticeError::Message("GGUF tensor data offset overflowed".to_string()))
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.offset.checked_add(len).ok_or_else(|| {
            LatticeError::Message("GGUF parser overflowed while advancing cursor".to_string())
        })?;
        let slice = self.bytes.get(self.offset..end).ok_or_else(|| {
            LatticeError::Message(format!(
                "unexpected end of GGUF file while reading {len} byte(s) at offset {}",
                self.offset
            ))
        })?;
        self.offset = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let bytes = self.read_exact(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let bytes = self.read_exact(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let bytes = self.read_exact(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_bool(&mut self) -> Result<bool> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(LatticeError::Message(format!(
                "invalid GGUF boolean value: {value}"
            ))),
        }
    }

    fn read_string(&mut self) -> Result<String> {
        let len = usize::try_from(self.read_u64()?).map_err(|_| {
            LatticeError::Message("GGUF string length does not fit into usize".to_string())
        })?;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|error| {
            LatticeError::Message(format!("GGUF string value is not valid UTF-8: {error}"))
        })
    }

    fn read_value(&mut self, value_type: GgufMetadataValueType) -> Result<GgufMetadataValue> {
        let value = match value_type {
            GgufMetadataValueType::U8 => GgufMetadataValue::U8(self.read_u8()?),
            GgufMetadataValueType::I8 => GgufMetadataValue::I8(self.read_i8()?),
            GgufMetadataValueType::U16 => GgufMetadataValue::U16(self.read_u16()?),
            GgufMetadataValueType::I16 => GgufMetadataValue::I16(self.read_i16()?),
            GgufMetadataValueType::U32 => GgufMetadataValue::U32(self.read_u32()?),
            GgufMetadataValueType::I32 => GgufMetadataValue::I32(self.read_i32()?),
            GgufMetadataValueType::F32 => GgufMetadataValue::F32(self.read_f32()?),
            GgufMetadataValueType::Bool => GgufMetadataValue::Bool(self.read_bool()?),
            GgufMetadataValueType::String => GgufMetadataValue::String(self.read_string()?),
            GgufMetadataValueType::Array => {
                let element_type = GgufMetadataValueType::from_raw(self.read_u32()?)?;
                if element_type == GgufMetadataValueType::Array {
                    return Err(LatticeError::Message(
                        "nested GGUF metadata arrays are not supported".to_string(),
                    ));
                }

                let len = usize::try_from(self.read_u64()?).map_err(|_| {
                    LatticeError::Message(
                        "GGUF metadata array length does not fit into usize".to_string(),
                    )
                })?;
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_value(element_type)?);
                }
                GgufMetadataValue::Array(values)
            }
            GgufMetadataValueType::U64 => GgufMetadataValue::U64(self.read_u64()?),
            GgufMetadataValueType::I64 => GgufMetadataValue::I64(self.read_i64()?),
            GgufMetadataValueType::F64 => GgufMetadataValue::F64(self.read_f64()?),
        };

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArchitectureMetadata, GgmlType, GgufMetadataValue, GgufMetadataValueType, LlamaMetadata,
        parse_gguf, parse_gguf_metadata,
    };

    #[test]
    fn parses_generic_metadata_and_llama_view() {
        let bytes = build_gguf(
            &[
                kv_string("general.architecture", "llama"),
                kv_string("general.name", "TinyLlama"),
                kv_u32("general.alignment", 32),
                kv_u32("llama.context_length", 2048),
                kv_u32("llama.embedding_length", 128),
                kv_u32("llama.block_count", 4),
                kv_u32("llama.feed_forward_length", 512),
                kv_u32("llama.attention.head_count", 8),
                kv_u32("llama.attention.head_count_kv", 4),
                MetadataEntry::new("llama.rope.dimension_count", TestValue::U32(64)),
                MetadataEntry::new("llama.rope.freq_base", TestValue::F32(10000.0)),
                MetadataEntry::new(
                    "tokenizer.ggml.tokens",
                    TestValue::StringArray(vec!["<s>", "hello"]),
                ),
            ],
            &[TensorEntry {
                name: "blk.0.attn_norm.weight",
                dimensions: vec![128],
                ggml_type: 0,
                offset: 0,
                data: vec![],
            }],
            32,
        );
        let metadata = parse_gguf_metadata(&bytes).expect("GGUF metadata should parse");

        assert_eq!(metadata.header().version(), 3);
        assert_eq!(metadata.header().tensor_count(), 1);
        assert_eq!(metadata.architecture_name(), Some("llama"));
        assert_eq!(metadata.alignment().expect("alignment"), 32);
        assert_eq!(
            metadata.metadata_value("general.name"),
            Some(&GgufMetadataValue::String("TinyLlama".to_string()))
        );
        assert_eq!(metadata.tensor_infos().len(), 1);
        assert_eq!(metadata.tensor_infos()[0].name(), "blk.0.attn_norm.weight");
        assert_eq!(metadata.tensor_infos()[0].dimensions(), &[128]);
        assert_eq!(metadata.tensor_infos()[0].ggml_type(), 0);
        assert_eq!(metadata.tensor_data_offset() % 32, 0);
        assert_eq!(
            metadata.tensor_infos()[0]
                .file_offset(metadata.tensor_data_offset())
                .expect("tensor file offset"),
            metadata.tensor_data_offset()
        );

        let ArchitectureMetadata::Llama(llama) = metadata
            .architecture_metadata()
            .expect("llama architecture metadata should exist")
        else {
            panic!("expected llama architecture metadata");
        };

        assert_eq!(
            llama,
            LlamaMetadata {
                model_name: Some("TinyLlama".to_string()),
                context_length: Some(2048),
                embedding_length: Some(128),
                block_count: Some(4),
                feed_forward_length: Some(512),
                attention_head_count: Some(8),
                attention_head_count_kv: Some(4),
                rope_dimension_count: Some(64),
                rope_freq_base: Some(10000.0),
                vocab_size: Some(2),
            }
        );
    }

    #[test]
    fn defaults_alignment_when_metadata_is_missing() {
        let bytes = build_gguf(
            &[kv_string("general.architecture", "llama")],
            &[TensorEntry {
                name: "token_embd.weight",
                dimensions: vec![64, 128],
                ggml_type: 0,
                offset: 0,
                data: vec![],
            }],
            32,
        );

        let metadata = parse_gguf_metadata(&bytes).expect("GGUF metadata should parse");
        assert_eq!(metadata.alignment().expect("alignment"), 32);
        assert_eq!(metadata.tensor_data_offset() % 32, 0);
    }

    #[test]
    fn returns_unknown_architecture_metadata_for_untyped_architectures() {
        let bytes = build_gguf(
            &[kv_string("general.architecture", "qwen3")],
            &[TensorEntry {
                name: "blk.0.attn_q.weight",
                dimensions: vec![128, 128],
                ggml_type: 0,
                offset: 0,
                data: vec![],
            }],
            32,
        );

        let metadata = parse_gguf_metadata(&bytes).expect("GGUF metadata should parse");
        let architecture = metadata
            .architecture_metadata()
            .expect("architecture metadata should exist");

        assert_eq!(
            architecture,
            ArchitectureMetadata::Unknown("qwen3".to_string())
        );
    }

    #[test]
    fn rejects_non_power_of_two_alignment() {
        let bytes = build_gguf(
            &[
                kv_string("general.architecture", "llama"),
                kv_u32("general.alignment", 24),
            ],
            &[TensorEntry {
                name: "output.weight",
                dimensions: vec![128, 128],
                ggml_type: 0,
                offset: 0,
                data: vec![],
            }],
            24,
        );

        let error =
            parse_gguf_metadata(&bytes).expect_err("non-power-of-two alignment should fail");
        assert!(
            error
                .to_string()
                .contains("GGUF alignment must be a power of two"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn supports_tensor_name_lookup_and_zero_copy_views() {
        let first_tensor = vec![0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64];
        let second_tensor = (0_u8..34).collect::<Vec<_>>();
        let bytes = build_gguf(
            &[kv_string("general.architecture", "llama")],
            &[
                TensorEntry {
                    name: "token_embd.weight",
                    dimensions: vec![4],
                    ggml_type: GgmlType::F32.raw(),
                    offset: 0,
                    data: first_tensor.clone(),
                },
                TensorEntry {
                    name: "blk.0.attn_q.weight",
                    dimensions: vec![32],
                    ggml_type: GgmlType::Q8_0.raw(),
                    offset: 32,
                    data: second_tensor.clone(),
                },
            ],
            32,
        );

        let gguf = parse_gguf(&bytes).expect("GGUF file should parse");
        let metadata = gguf.metadata();
        assert!(metadata.tensor_info("missing.weight").is_none());

        let info = metadata
            .tensor_info("blk.0.attn_q.weight")
            .expect("tensor info should exist");
        assert_eq!(info.parsed_ggml_type().expect("ggml type"), GgmlType::Q8_0);
        assert_eq!(info.byte_len().expect("byte len"), 34);
        assert_eq!(info.element_count().expect("element count"), 32);

        let first_view = gguf
            .tensor("token_embd.weight")
            .expect("tensor lookup should succeed")
            .expect("tensor view should exist");
        assert_eq!(first_view.ggml_type().expect("ggml type"), GgmlType::F32);
        assert_eq!(first_view.byte_len(), 16);
        assert_eq!(first_view.data(), first_tensor.as_slice());
        assert_eq!(
            first_view.file_range(),
            metadata.tensor_data_offset()..metadata.tensor_data_offset() + 16
        );

        let second_view = gguf
            .tensor("blk.0.attn_q.weight")
            .expect("tensor lookup should succeed")
            .expect("tensor view should exist");
        assert_eq!(second_view.ggml_type().expect("ggml type"), GgmlType::Q8_0);
        assert_eq!(second_view.byte_len(), 34);
        assert_eq!(second_view.data(), second_tensor.as_slice());
        assert_eq!(
            second_view.file_range(),
            metadata.tensor_data_offset() + 32..metadata.tensor_data_offset() + 66
        );
    }

    #[test]
    fn rejects_invalid_magic() {
        let error = parse_gguf_metadata(b"NOTGGUF").expect_err("invalid magic should fail");
        assert!(
            error.to_string().contains("invalid GGUF magic"),
            "unexpected error: {error}"
        );
    }

    fn build_gguf(
        metadata_entries: &[MetadataEntry<'_>],
        tensor_entries: &[TensorEntry<'_>],
        alignment: usize,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, tensor_entries.len() as u64);
        push_u64(&mut bytes, metadata_entries.len() as u64);

        for entry in metadata_entries {
            push_string(&mut bytes, entry.key);
            push_value(&mut bytes, &entry.value);
        }

        for tensor in tensor_entries {
            push_string(&mut bytes, tensor.name);
            push_u32(&mut bytes, tensor.dimensions.len() as u32);
            for dimension in &tensor.dimensions {
                push_u64(&mut bytes, *dimension);
            }
            push_u32(&mut bytes, tensor.ggml_type);
            push_u64(&mut bytes, tensor.offset);
        }

        while bytes.len() % alignment != 0 {
            bytes.push(0);
        }

        let mut tensor_data = vec![0_u8; alignment.max(1)];
        for tensor in tensor_entries {
            let offset = tensor.offset as usize;
            let end = offset + tensor.data.len();
            if tensor_data.len() < end {
                tensor_data.resize(end, 0);
            }
            tensor_data[offset..end].copy_from_slice(&tensor.data);
        }

        bytes.extend_from_slice(&tensor_data);
        bytes
    }

    fn kv_string<'a>(key: &'a str, value: &'a str) -> MetadataEntry<'a> {
        MetadataEntry::new(key, TestValue::String(value))
    }

    fn kv_u32<'a>(key: &'a str, value: u32) -> MetadataEntry<'a> {
        MetadataEntry::new(key, TestValue::U32(value))
    }

    fn push_value(bytes: &mut Vec<u8>, value: &TestValue<'_>) {
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

    struct TensorEntry<'a> {
        name: &'a str,
        dimensions: Vec<u64>,
        ggml_type: u32,
        offset: u64,
        data: Vec<u8>,
    }

    struct MetadataEntry<'a> {
        key: &'a str,
        value: TestValue<'a>,
    }

    impl<'a> MetadataEntry<'a> {
        fn new(key: &'a str, value: TestValue<'a>) -> Self {
            Self { key, value }
        }
    }

    enum TestValue<'a> {
        String(&'a str),
        U32(u32),
        F32(f32),
        StringArray(Vec<&'a str>),
    }
}
