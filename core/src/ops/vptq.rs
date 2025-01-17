use crate::{internal::*, ops::array::GatherElements, ops::einsum::EinSum};

#[derive(Debug, Clone)]
pub struct VPTQGemm {
    pub vector_len: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl Op for VPTQGemm {
    fn name(&self) -> Cow<str> {
        "VPTQGemm".into()
    }

    op_as_typed_op!();
}

impl VPTQGemm {
    fn eval_extract_from_vector_quant(
        &self,
        centroids: Tensor,
        indices: Tensor,
    ) -> TractResult<Tensor> {
        let mut indices = indices.clone();
        let [num_codebooks, num_centroids, vector_len] = *centroids.shape() else {
            unimplemented!("unexected centroid shape ?")
        };

        let [_, _, group_size] = *centroids.shape() else {
            unimplemented!("unexected indice shape ?")
        };

        let mut vsh = indices.shape().to_vec();
        indices.insert_axis(3)?;
        vsh.push(vector_len);
        indices = indices.broadcast_to_shape(&vsh)?;
        let intermediate_volume = indices.shape()[1..3].iter().fold(1, |r, x| r * x);
        indices = indices.into_shape(&[num_codebooks, intermediate_volume, vector_len])?;

        let gather1 = GatherElements { axis: 1 };
        // selected_centroids = torch.gather(centroids, 1, indices)
        let selected_centroids = gather1
            .eval(tvec!(centroids.into(), indices.into()))?
            .pop()
            .context("apply gather to get selected main centroids")
            .unwrap()
            .into_tensor();

        let remain = selected_centroids.volume() / (num_codebooks * group_size * vector_len);
        let mut qweight = selected_centroids
            .into_shape(&[num_codebooks, remain, group_size, vector_len])?
            .permute_axes(&[0, 1, 3, 2])?
            .into_shape(&[num_codebooks, remain * vector_len, group_size])?
            .permute_axes(&[1, 0, 2])?
            .into_shape(&[vector_len * remain, num_codebooks * group_size])?;

        let dim0 = qweight.shape()[0];
        let padding = (-(self.out_features as i16) % vector_len as i16) as usize;
        if padding > 0 {
            qweight = qweight.slice(0, 0, dim0 - padding)?;
        }
        Ok(qweight)
    }
}

impl EvalOp for VPTQGemm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (
            input,
            indices,
            centroids,
            outlier_indices,
            outlier_centroids,
            perm,
            weight_scale,
            weight_bias,
            bias,
        ) = args_9!(inputs);
        let mut indices = indices.into_tensor();
        let mut centroids = centroids.into_tensor();
        let mut outlier_indices = outlier_indices.into_tensor();
        let mut outlier_centroids = outlier_centroids.into_tensor();
        let mut perm = perm.into_tensor();
        let mut weight_scale = weight_scale.into_tensor();
        let mut weight_bias = weight_bias.into_tensor();
        let mut bias = bias.into_tensor();

        if weight_scale.len() > 1 {
            unimplemented!("'weight scale' for vptq not yet supported !");
        }
        if weight_bias.len() > 1 {
            unimplemented!("'weight bias' for vptq not yet supported !");
        }
        let enable_norm = weight_scale.len() > 1 && weight_bias.len() > 1;
        if bias.len() > 1 {
            unimplemented!("'bias' for vptq not yet supported !");
        }
        assert_eq!(input.rank(), 3);
        assert!(input.datum_type().is_float());

        assert_eq!(indices.rank(), 3);
        assert_eq!(indices.datum_type(), DatumType::U16);
        assert_eq!(centroids.rank(), 3);
        assert!(centroids.datum_type().is_float());

        let enable_outlier = outlier_indices.len() > 0;
        if enable_outlier {
            assert_eq!(outlier_indices.rank(), 3);
            assert_eq!(outlier_indices.datum_type(), DatumType::U16);
            assert_eq!(outlier_centroids.rank(), 3);
            assert!(outlier_centroids.datum_type().is_float());
        }

        let mut qweight = self.eval_extract_from_vector_quant(centroids, indices)?;
        if enable_outlier {
            // same as centroids to qweights except for outlier
            let outlier_qweight =
                self.eval_extract_from_vector_quant(outlier_centroids, outlier_indices)?;
            // qweight = torch.cat([qweight_outlier, qweight], dim=1)
            qweight = Tensor::stack_tensors(0, &[qweight, outlier_qweight])?;
        }

        // let enable_perm = perm.len() <= 1;
        // if enable_perm {
        //     // TODO: manage case with packed indice
        //     //     if self.is_indice_packed:
        //     //         invert_perm = torch.argsort(
        //     //             self.perm.to(torch.uint16).to(torch.int64)
        //     //         )
        //     //     else:
        //     //         invert_perm = torch.argsort(self.perm)
        //     //     if self.vector_quant_dim == "in":
        //     //         assert True, "Not implemented"
        //     //         # qweight = qweight[invert_perm, :]
        //     //     else:
        //     //         qweight = qweight[:, invert_perm]
        //     let invert_perm = perm.into_tensor().argsort();
        // }

        if enable_norm {
            qweight = (qweight.into_array::<f32>()? * weight_scale.into_array::<f32>()?
                + weight_bias.into_array::<f32>()?)
            .into_tensor();
        }
        // call matmul now with qweight

        let einsum_op = EinSum::new("bik,kj->ij".parse()?, f32::datum_type());
        einsum_op.eval(tvec!(input, qweight.into_tvalue()))
    }
}

impl TypedOp for VPTQGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].without_value()))
    }

    as_op!();
}
