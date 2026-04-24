use half::f16;
use lattice_core::{LatticeError, Result};
use lattice_model::{GgmlType, GgufTensorView};

use super::tensor::{row_bytes, tensor_inner_dim, tensor_row_count};

pub(crate) fn linear_batch(inputs: &[Vec<f32>], weight: &GgufTensorView<'_>) -> Result<Vec<Vec<f32>>> {
    inputs.iter().map(|input| linear(input, weight)).collect()
}

pub(crate) fn linear(input: &[f32], weight: &GgufTensorView<'_>) -> Result<Vec<f32>> {
    let inner_dim = tensor_inner_dim(weight)?;
    if input.len() != inner_dim {
        return Err(LatticeError::Message(format!(
            "linear input width {} does not match tensor `{}` inner dim {}",
            input.len(),
            weight.name(),
            inner_dim
        )));
    }

    let row_count = tensor_row_count(weight)?;
    let mut output = Vec::with_capacity(row_count);
    for row_index in 0..row_count {
        output.push(dot_row(weight, row_index, input)?);
    }

    Ok(output)
}

pub(crate) fn dot(lhs: &[f32], rhs: &[f32]) -> Result<f32> {
    if lhs.len() != rhs.len() {
        return Err(LatticeError::Message(format!(
            "dot width mismatch: lhs {}, rhs {}",
            lhs.len(),
            rhs.len()
        )));
    }

    Ok(lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| lhs * rhs).sum())
}

fn dot_row(tensor: &GgufTensorView<'_>, row_index: usize, input: &[f32]) -> Result<f32> {
    let row = row_bytes(tensor, row_index)?;
    let inner_dim = tensor_inner_dim(tensor)?;
    if input.len() != inner_dim {
        return Err(LatticeError::Message(format!(
            "dot input width {} does not match tensor `{}` inner dim {}",
            input.len(),
            tensor.name(),
            inner_dim
        )));
    }

    match tensor.ggml_type()? {
        GgmlType::F32 => Ok(row
            .chunks_exact(4)
            .zip(input.iter())
            .map(|(chunk, input)| {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) * input
            })
            .sum()),
        GgmlType::F16 => Ok(row
            .chunks_exact(2)
            .zip(input.iter())
            .map(|(chunk, input)| {
                f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32() * input
            })
            .sum()),
        GgmlType::Q8_0 => dot_q8_0_row(row, input),
        ggml_type => Err(LatticeError::Message(format!(
            "matmul does not support {:?} for tensor `{}`",
            ggml_type,
            tensor.name()
        ))),
    }
}

fn dot_q8_0_row(row: &[u8], input: &[f32]) -> Result<f32> {
    if !input.len().is_multiple_of(32) {
        return Err(LatticeError::Message(format!(
            "q8_0 input width {} is not divisible by 32",
            input.len()
        )));
    }

    let mut sum = 0.0_f32;
    for (block_index, block) in row.chunks_exact(34).enumerate() {
        let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let input_offset = block_index * 32;
        for (index, quantized) in block[2..34].iter().enumerate() {
            sum += scale * (*quantized as i8 as f32) * input[input_offset + index];
        }
    }

    Ok(sum)
}

#[cfg(test)]
mod tests {
    use lattice_model::parse_gguf;

    use super::{dot, linear};

    #[test]
    fn computes_dense_linear() {
        let bytes = build_f32_matrix_gguf();
        let gguf = parse_gguf(&bytes).expect("GGUF should parse");
        let tensor = gguf
            .tensor("weight")
            .expect("tensor lookup should succeed")
            .expect("tensor should exist");

        let output = linear(&[2.0, 3.0], &tensor).expect("linear should work");
        assert_eq!(output, vec![5.0, 6.0]);
    }

    #[test]
    fn computes_dot_product() {
        let output = dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).expect("dot should work");
        assert_eq!(output, 32.0);
    }

    fn build_f32_matrix_gguf() -> Vec<u8> {
        build_gguf(
            &[MetadataEntry::string("general.architecture", "llama")],
            &[TensorEntry::f32("weight", vec![2, 2], &[1.0, 1.0, 0.0, 2.0])],
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
