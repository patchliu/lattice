use half::f16;
use lattice_core::{LatticeError, Result};
use lattice_model::{GgmlType, GgufTensorView};

pub(crate) fn decode_vector(tensor: &GgufTensorView<'_>) -> Result<Vec<f32>> {
    let dimensions = tensor.dimensions();
    if dimensions.len() != 1 {
        return Err(LatticeError::Message(format!(
            "expected vector tensor `{}` but found shape {:?}",
            tensor.name(),
            dimensions
        )));
    }

    decode_row(tensor, 0)
}

pub(crate) fn decode_row(tensor: &GgufTensorView<'_>, row_index: usize) -> Result<Vec<f32>> {
    let row = row_bytes(tensor, row_index)?;
    let inner_dim = tensor_inner_dim(tensor)?;
    let mut values = vec![0.0; inner_dim];

    match tensor.ggml_type()? {
        GgmlType::F32 => {
            for (output, chunk) in values.iter_mut().zip(row.chunks_exact(4)) {
                *output = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GgmlType::F16 => {
            for (output, chunk) in values.iter_mut().zip(row.chunks_exact(2)) {
                *output = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            }
        }
        GgmlType::Q8_0 => {
            decode_q8_0_row(row, &mut values)?;
        }
        ggml_type => {
            return Err(LatticeError::Message(format!(
                "tensor decode does not support {:?} for tensor `{}`",
                ggml_type,
                tensor.name()
            )));
        }
    }

    Ok(values)
}

pub(crate) fn row_bytes<'a>(tensor: &'a GgufTensorView<'a>, row_index: usize) -> Result<&'a [u8]> {
    let row_count = tensor_row_count(tensor)?;
    if row_index >= row_count {
        return Err(LatticeError::Message(format!(
            "row index {} is out of range for tensor `{}` with {} row(s)",
            row_index,
            tensor.name(),
            row_count
        )));
    }

    let inner_dim = tensor_inner_dim(tensor)?;
    let layout = tensor.ggml_type()?.layout();
    let row_bytes = inner_dim
        .checked_div(layout.block_size())
        .and_then(|blocks| blocks.checked_mul(layout.bytes_per_block()))
        .ok_or_else(|| {
            LatticeError::Message(format!(
                "row byte length overflowed for tensor `{}`",
                tensor.name()
            ))
        })?;
    let start = row_index.checked_mul(row_bytes).ok_or_else(|| {
        LatticeError::Message(format!(
            "row offset overflowed for tensor `{}`",
            tensor.name()
        ))
    })?;
    let end = start.checked_add(row_bytes).ok_or_else(|| {
        LatticeError::Message(format!(
            "row range overflowed for tensor `{}`",
            tensor.name()
        ))
    })?;

    tensor.data().get(start..end).ok_or_else(|| {
        LatticeError::Message(format!(
            "row range {}..{} is out of bounds for tensor `{}` with {} bytes",
            start,
            end,
            tensor.name(),
            tensor.byte_len()
        ))
    })
}

pub(crate) fn tensor_inner_dim(tensor: &GgufTensorView<'_>) -> Result<usize> {
    let inner_dim = tensor.dimensions().first().copied().unwrap_or(1);
    usize::try_from(inner_dim).map_err(|_| {
        LatticeError::Message(format!(
            "tensor `{}` inner dimension does not fit into usize",
            tensor.name()
        ))
    })
}

pub(crate) fn tensor_row_count(tensor: &GgufTensorView<'_>) -> Result<usize> {
    tensor
        .dimensions()
        .iter()
        .skip(1)
        .try_fold(1_usize, |rows, dimension| {
            let dimension = usize::try_from(*dimension).map_err(|_| {
                LatticeError::Message(format!(
                    "tensor `{}` row dimension does not fit into usize",
                    tensor.name()
                ))
            })?;
            rows.checked_mul(dimension).ok_or_else(|| {
                LatticeError::Message(format!("tensor `{}` row count overflowed", tensor.name()))
            })
        })
}

fn decode_q8_0_row(row: &[u8], output: &mut [f32]) -> Result<()> {
    let blocks = output.len() / 32;
    for (block_index, block) in row.chunks_exact(34).enumerate() {
        if block_index >= blocks {
            break;
        }

        let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let values = &block[2..34];
        let output_offset = block_index * 32;
        for (index, quantized) in values.iter().enumerate() {
            output[output_offset + index] = scale * (*quantized as i8 as f32);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use lattice_model::parse_gguf;

    use super::{decode_row, decode_vector};

    #[test]
    fn decodes_f32_vector() {
        let bytes = build_f32_tensor_gguf();
        let gguf = parse_gguf(&bytes).expect("GGUF should parse");
        let tensor = gguf
            .tensor("vector.weight")
            .expect("tensor lookup should succeed")
            .expect("tensor should exist");

        let values = decode_vector(&tensor).expect("vector should decode");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn decodes_q8_0_row() {
        let bytes = build_q8_tensor_gguf();
        let gguf = parse_gguf(&bytes).expect("GGUF should parse");
        let tensor = gguf
            .tensor("matrix.weight")
            .expect("tensor lookup should succeed")
            .expect("tensor should exist");

        let row = decode_row(&tensor, 0).expect("row should decode");
        assert_eq!(row.len(), 32);
        assert_eq!(row[0], 1.0);
        assert_eq!(row[1], -2.0);
        assert_eq!(row[2], 3.0);
    }

    fn build_f32_tensor_gguf() -> Vec<u8> {
        build_gguf(
            &[MetadataEntry::string("general.architecture", "llama")],
            &[TensorEntry::f32("vector.weight", vec![4], &[1.0, 2.0, 3.0, 4.0])],
            32,
        )
    }

    fn build_q8_tensor_gguf() -> Vec<u8> {
        let mut q8_data = Vec::with_capacity(34);
        q8_data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        let mut quantized = [0_i8; 32];
        quantized[0] = 1;
        quantized[1] = -2;
        quantized[2] = 3;
        q8_data.extend(quantized.iter().map(|value| *value as u8));

        build_gguf(
            &[MetadataEntry::string("general.architecture", "llama")],
            &[TensorEntry {
                name: "matrix.weight".to_string(),
                dimensions: vec![32, 1],
                ggml_type: lattice_model::GgmlType::Q8_0.raw(),
                data: q8_data,
            }],
            32,
        )
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
            push_u32(&mut bytes, 8);
            push_string(&mut bytes, &entry.value);
        }

        let mut next_offset = 0_usize;
        let mut offsets = Vec::with_capacity(tensor_entries.len());
        for tensor in tensor_entries {
            next_offset = align_to(next_offset, alignment);
            offsets.push(next_offset as u64);
            next_offset += tensor.data.len();
        }

        for (tensor, offset) in tensor_entries.iter().zip(offsets.iter()) {
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
        for (tensor, offset) in tensor_entries.iter().zip(offsets.iter()) {
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

    struct MetadataEntry {
        key: String,
        value: String,
    }

    impl MetadataEntry {
        fn string(key: &str, value: &str) -> Self {
            Self {
                key: key.to_string(),
                value: value.to_string(),
            }
        }
    }

    struct TensorEntry {
        name: String,
        dimensions: Vec<u64>,
        ggml_type: u32,
        data: Vec<u8>,
    }

    impl TensorEntry {
        fn f32(name: &str, dimensions: Vec<u64>, values: &[f32]) -> Self {
            let mut data = Vec::with_capacity(values.len() * 4);
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }

            Self {
                name: name.to_string(),
                dimensions,
                ggml_type: lattice_model::GgmlType::F32.raw(),
                data,
            }
        }
    }
}
