use lattice_core::{LatticeError, Result};

pub(crate) fn apply_rope_in_place(
    vectors: &mut [Vec<f32>],
    head_count: usize,
    head_width: usize,
    rope_dimension_count: usize,
    rope_freq_base: f64,
) -> Result<()> {
    if rope_dimension_count > head_width {
        return Err(LatticeError::Message(format!(
            "rope dimension {} exceeds head width {}",
            rope_dimension_count, head_width
        )));
    }

    let rotate_width = rope_dimension_count - (rope_dimension_count % 2);
    for (position, vector) in vectors.iter_mut().enumerate() {
        if vector.len() != head_count * head_width {
            return Err(LatticeError::Message(format!(
                "rope input width {} does not match expected {}",
                vector.len(),
                head_count * head_width
            )));
        }

        for head_index in 0..head_count {
            let head_offset = head_index * head_width;
            for pair_offset in (0..rotate_width).step_by(2) {
                let exponent = pair_offset as f64 / rope_dimension_count as f64;
                let angle = position as f64 / rope_freq_base.powf(exponent);
                let cos = angle.cos() as f32;
                let sin = angle.sin() as f32;
                let first_index = head_offset + pair_offset;
                let second_index = first_index + 1;
                let first = vector[first_index];
                let second = vector[second_index];
                vector[first_index] = first * cos - second * sin;
                vector[second_index] = first * sin + second * cos;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::apply_rope_in_place;

    #[test]
    fn rotates_second_position() {
        let mut vectors = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        apply_rope_in_place(&mut vectors, 1, 2, 2, 10_000.0).expect("rope should work");
        assert_eq!(vectors[0], vec![1.0, 0.0]);
        assert!(vectors[1][0] > 0.54 && vectors[1][0] < 0.55);
        assert!(vectors[1][1] > 0.84 && vectors[1][1] < 0.85);
    }
}
