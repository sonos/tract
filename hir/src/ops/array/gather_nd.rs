use crate::internal::*;
pub use tract_core::ops::array::GatherNd;

impl InferenceRulesOp for GatherNd {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given(&inputs[1].rank, move |s, indices_rank| {
            let indices_rank = indices_rank as usize;
            for i in 0..(indices_rank - 1) {
                s.equals(&outputs[0].shape[i], &inputs[1].shape[i])?;
            }
            s.given_2(
                &inputs[1].shape[indices_rank - 1],
                &inputs[0].rank,
                move |s, n, data_rank| {
                    if let Ok(n) = n.to_i64() {
                        let n = n as usize + self.batch_dims;
                        ensure!(
                            n <= data_rank as usize,
                            "GatherNd indices index {n} axes (last indices dimension plus batch_dims) but data has rank {data_rank}"
                        );
                        s.equals(
                            &outputs[0].rank,
                            (indices_rank - 1 + data_rank as usize - n) as i64,
                        )?;
                        for i in 0..(data_rank as usize - n) {
                            s.equals(
                                &outputs[0].shape[indices_rank - 1 + i],
                                &inputs[0].shape[n + i],
                            )?;
                        }
                    }
                    Ok(())
                },
            )
        })
    }

    as_op!();
    to_typed!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn infer(
        batch_dims: usize,
        data: InferenceFact,
        indices: InferenceFact,
    ) -> TractResult<TVec<InferenceFact>> {
        let mut op = GatherNd::new(batch_dims);
        let output = InferenceFact::default();
        Ok(op.infer_facts(tvec!(&data, &indices), tvec!(&output), tvec!())?.1)
    }

    #[test]
    fn trailing_dims_come_from_data() {
        let facts = infer(0, f32::fact([8, 128, 256]).into(), i64::fact([32, 1]).into()).unwrap();
        assert_eq!(facts, tvec!(f32::fact([32, 128, 256]).into()));
    }

    #[test]
    fn batch_dims_shift_the_data_axes() {
        let facts = infer(1, f32::fact([4, 15, 18]).into(), i64::fact([4, 15, 1]).into()).unwrap();
        assert_eq!(facts, tvec!(f32::fact([4, 15, 18]).into()));
    }

    #[test]
    fn indexing_every_data_axis_leaves_the_index_prefix() {
        let facts = infer(0, f32::fact([4, 5, 6]).into(), i64::fact([7, 3]).into()).unwrap();
        assert_eq!(facts, tvec!(f32::fact([7]).into()));
    }

    #[test]
    fn indices_deeper_than_data_rank_is_rejected() {
        infer(0, f32::fact([4, 5, 6]).into(), i64::fact([7, 4]).into()).unwrap_err();
    }
}
