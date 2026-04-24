use lattice_core::{LatticeError, Result};

use super::matmul::dot;

pub(crate) fn causal_attention(
    queries: &[Vec<f32>],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    attention_head_count: usize,
    attention_head_count_kv: usize,
    head_dimension: usize,
    value_dimension: usize,
) -> Result<Vec<Vec<f32>>> {
    if queries.len() != keys.len() || queries.len() != values.len() {
        return Err(LatticeError::Message(format!(
            "attention batch mismatch: q {}, k {}, v {}",
            queries.len(),
            keys.len(),
            values.len()
        )));
    }
    if !attention_head_count.is_multiple_of(attention_head_count_kv) {
        return Err(LatticeError::Message(format!(
            "attention head count {} is not divisible by kv head count {}",
            attention_head_count, attention_head_count_kv
        )));
    }

    let kv_repeat = attention_head_count / attention_head_count_kv;
    let score_scale = 1.0 / (head_dimension as f32).sqrt();
    let mut outputs = vec![vec![0.0; attention_head_count * value_dimension]; queries.len()];

    for token_index in 0..queries.len() {
        for head_index in 0..attention_head_count {
            let kv_head_index = head_index / kv_repeat;
            let query = head_slice(&queries[token_index], head_index, head_dimension)?;
            let mut scores = Vec::with_capacity(token_index + 1);
            for past_index in 0..=token_index {
                let key = head_slice(&keys[past_index], kv_head_index, head_dimension)?;
                scores.push(dot(query, key)? * score_scale);
            }
            let probabilities = softmax(&scores);

            let output_head =
                mutable_head_slice(&mut outputs[token_index], head_index, value_dimension)?;
            for (past_index, probability) in probabilities.iter().enumerate() {
                let value = head_slice(&values[past_index], kv_head_index, value_dimension)?;
                for (output, value) in output_head.iter_mut().zip(value.iter()) {
                    *output += probability * value;
                }
            }
        }
    }

    Ok(outputs)
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = scores
        .iter()
        .map(|score| (*score - max_score).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    exps.into_iter().map(|value| value / sum).collect()
}

fn head_slice(vector: &[f32], head_index: usize, head_width: usize) -> Result<&[f32]> {
    let start = head_index
        .checked_mul(head_width)
        .ok_or_else(|| LatticeError::Message("head slice start overflowed".to_string()))?;
    let end = start
        .checked_add(head_width)
        .ok_or_else(|| LatticeError::Message("head slice end overflowed".to_string()))?;
    vector.get(start..end).ok_or_else(|| {
        LatticeError::Message(format!(
            "head slice {}..{} exceeds vector width {}",
            start,
            end,
            vector.len()
        ))
    })
}

fn mutable_head_slice(
    vector: &mut [f32],
    head_index: usize,
    head_width: usize,
) -> Result<&mut [f32]> {
    let vector_len = vector.len();
    let start = head_index
        .checked_mul(head_width)
        .ok_or_else(|| LatticeError::Message("mutable head slice start overflowed".to_string()))?;
    let end = start
        .checked_add(head_width)
        .ok_or_else(|| LatticeError::Message("mutable head slice end overflowed".to_string()))?;
    vector.get_mut(start..end).ok_or_else(|| {
        LatticeError::Message(format!(
            "mutable head slice {}..{} exceeds vector width {}",
            start, end, vector_len
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::causal_attention;

    #[test]
    fn attends_causally() {
        let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let keys = queries.clone();
        let values = vec![vec![2.0, 0.0], vec![0.0, 4.0]];

        let outputs =
            causal_attention(&queries, &keys, &values, 1, 1, 2, 2).expect("attention should work");

        assert_eq!(outputs[0], vec![2.0, 0.0]);
        assert!(outputs[1][0] > 0.0);
        assert!(outputs[1][1] > outputs[1][0]);
    }
}
