use lattice_core::{LatticeError, Result};
use lattice_model::{LlamaBlockTensors, LlamaModel};

use crate::LoadedLlama;
use crate::ops::{
    apply_rope_in_place, causal_attention, decode_row, decode_vector, linear, linear_batch,
    rms_norm, rms_norm_batch, swiglu_batch,
};

#[cfg(test)]
mod tests;

/// Output produced by a full LLaMA prefill pass over a prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillOutput {
    /// The complete logits vector for the final prompt position.
    pub logits: Vec<f32>,
    /// The argmax token id for the final prompt position.
    pub next_token_id: u32,
}

impl<'a> LoadedLlama<'a> {
    /// Runs a full prefill pass over the provided token ids and returns final-position logits.
    pub fn prefill(&self, token_ids: &[u32]) -> Result<PrefillOutput> {
        prefill(self.model(), token_ids)
    }
}

pub(crate) fn prefill(model: &LlamaModel<'_>, token_ids: &[u32]) -> Result<PrefillOutput> {
    if token_ids.is_empty() {
        return Err(LatticeError::Message(
            "prefill requires at least one token id".to_string(),
        ));
    }
    if token_ids.len() > model.spec.context_length {
        return Err(LatticeError::Message(format!(
            "prompt length {} exceeds model context length {}",
            token_ids.len(),
            model.spec.context_length
        )));
    }

    let mut hidden_states = embed_tokens(model, token_ids)?;
    for block in &model.tensors.blocks {
        forward_block(model, block, &mut hidden_states)?;
    }

    let output_norm = decode_vector(&model.tensors.output_norm)?;
    let final_state = hidden_states
        .last()
        .ok_or_else(|| LatticeError::Message("prefill produced no hidden states".to_string()))?;
    let normalized = rms_norm(final_state, &output_norm, model.spec.rms_norm_epsilon)?;
    let logits = linear(&normalized, &model.tensors.output)?;
    let next_token_id = u32::try_from(argmax(&logits)).map_err(|_| {
        LatticeError::Message("argmax token id does not fit into u32".to_string())
    })?;

    Ok(PrefillOutput {
        logits,
        next_token_id,
    })
}

fn embed_tokens(model: &LlamaModel<'_>, token_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
    token_ids
        .iter()
        .map(|token_id| {
            let token_id = usize::try_from(*token_id).map_err(|_| {
                LatticeError::Message(format!("token id `{token_id}` does not fit into usize"))
            })?;
            if token_id >= model.spec.vocab_size {
                return Err(LatticeError::Message(format!(
                    "token id `{token_id}` is out of range for vocab size {}",
                    model.spec.vocab_size
                )));
            }

            decode_row(&model.tensors.token_embeddings, token_id)
        })
        .collect()
}

fn forward_block(
    model: &LlamaModel<'_>,
    block: &LlamaBlockTensors<'_>,
    hidden_states: &mut [Vec<f32>],
) -> Result<()> {
    let attention_norm = decode_vector(&block.attention_norm)?;
    let attention_input =
        rms_norm_batch(hidden_states, &attention_norm, model.spec.rms_norm_epsilon)?;
    let mut queries = linear_batch(&attention_input, &block.attention_q)?;
    let mut keys = linear_batch(&attention_input, &block.attention_k)?;
    let values = linear_batch(&attention_input, &block.attention_v)?;

    if model.spec.attention_key_length != model.spec.head_dimension {
        return Err(LatticeError::Message(format!(
            "attention key length {} does not match head dimension {}",
            model.spec.attention_key_length,
            model.spec.head_dimension
        )));
    }

    apply_rope_in_place(
        &mut queries,
        model.spec.attention_head_count,
        model.spec.head_dimension,
        model.spec.rope_dimension_count,
        model.spec.rope_freq_base,
    )?;
    apply_rope_in_place(
        &mut keys,
        model.spec.attention_head_count_kv,
        model.spec.attention_key_length,
        model.spec.rope_dimension_count,
        model.spec.rope_freq_base,
    )?;

    let attention = causal_attention(
        &queries,
        &keys,
        &values,
        model.spec.attention_head_count,
        model.spec.attention_head_count_kv,
        model.spec.head_dimension,
        model.spec.attention_value_length,
    )?;
    let attention_output = linear_batch(&attention, &block.attention_output)?;
    add_in_place(hidden_states, &attention_output)?;

    let ffn_norm = decode_vector(&block.ffn_norm)?;
    let ffn_input = rms_norm_batch(hidden_states, &ffn_norm, model.spec.rms_norm_epsilon)?;
    let gate = linear_batch(&ffn_input, &block.ffn_gate)?;
    let up = linear_batch(&ffn_input, &block.ffn_up)?;
    let ffn_hidden = swiglu_batch(&gate, &up)?;
    let ffn_output = linear_batch(&ffn_hidden, &block.ffn_down)?;
    add_in_place(hidden_states, &ffn_output)?;

    Ok(())
}

fn add_in_place(lhs: &mut [Vec<f32>], rhs: &[Vec<f32>]) -> Result<()> {
    if lhs.len() != rhs.len() {
        return Err(LatticeError::Message(format!(
            "residual add batch mismatch: lhs rows {}, rhs rows {}",
            lhs.len(),
            rhs.len()
        )));
    }

    for (lhs_row, rhs_row) in lhs.iter_mut().zip(rhs.iter()) {
        if lhs_row.len() != rhs_row.len() {
            return Err(LatticeError::Message(format!(
                "residual add width mismatch: lhs width {}, rhs width {}",
                lhs_row.len(),
                rhs_row.len()
            )));
        }

        for (lhs_value, rhs_value) in lhs_row.iter_mut().zip(rhs_row.iter()) {
            *lhs_value += *rhs_value;
        }
    }

    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_index = index;
            best_value = *value;
        }
    }
    best_index
}
