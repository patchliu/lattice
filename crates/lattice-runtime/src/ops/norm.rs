use lattice_core::{LatticeError, Result};

pub(crate) fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    if input.len() != weight.len() {
        return Err(LatticeError::Message(format!(
            "rms_norm width mismatch: input {}, weight {}",
            input.len(),
            weight.len()
        )));
    }

    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = 1.0 / (mean_square + epsilon).sqrt();
    Ok(input
        .iter()
        .zip(weight.iter())
        .map(|(value, weight)| value * inv_rms * weight)
        .collect())
}

pub(crate) fn rms_norm_batch(
    inputs: &[Vec<f32>],
    weights: &[f32],
    epsilon: f32,
) -> Result<Vec<Vec<f32>>> {
    inputs
        .iter()
        .map(|input| rms_norm(input, weights, epsilon))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{rms_norm, rms_norm_batch};

    #[test]
    fn normalizes_single_vector() {
        let output = rms_norm(&[3.0, 4.0], &[1.0, 1.0], 0.0).expect("norm should work");
        assert!(output[0] > 0.84 && output[0] < 0.85);
        assert!(output[1] > 1.13 && output[1] < 1.14);
    }

    #[test]
    fn normalizes_batch() {
        let output =
            rms_norm_batch(&[vec![1.0, 0.0], vec![0.0, 2.0]], &[1.0, 1.0], 1.0e-5).expect("norm should work");
        assert_eq!(output.len(), 2);
        assert!(output[0][0] > 1.41 && output[0][0] < 1.42);
        assert!(output[1][1] > 1.41 && output[1][1] < 1.42);
    }
}
