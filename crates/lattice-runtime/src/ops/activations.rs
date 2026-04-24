use lattice_core::{LatticeError, Result};

pub(crate) fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub(crate) fn swiglu_batch(gate: &[Vec<f32>], up: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    if gate.len() != up.len() {
        return Err(LatticeError::Message(format!(
            "swiglu batch mismatch: gate rows {}, up rows {}",
            gate.len(),
            up.len()
        )));
    }

    gate.iter()
        .zip(up.iter())
        .map(|(gate, up)| {
            if gate.len() != up.len() {
                return Err(LatticeError::Message(format!(
                    "swiglu width mismatch: gate width {}, up width {}",
                    gate.len(),
                    up.len()
                )));
            }

            Ok(gate
                .iter()
                .zip(up.iter())
                .map(|(gate, up)| silu(*gate) * *up)
                .collect())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{silu, swiglu_batch};

    #[test]
    fn computes_silu() {
        let value = silu(1.0);
        assert!(value > 0.73 && value < 0.74);
    }

    #[test]
    fn computes_swiglu_batch() {
        let output = swiglu_batch(&[vec![1.0, -1.0]], &[vec![2.0, 3.0]]).expect("swiglu should work");
        assert_eq!(output.len(), 1);
        assert!(output[0][0] > 1.46 && output[0][0] < 1.47);
        assert!(output[0][1] < -0.80 && output[0][1] > -0.81);
    }
}
