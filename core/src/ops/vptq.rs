use crate::{
    internal::*,
    ops::{
        array::{Gather, GatherElements, Topk},
        cast::cast,
        einsum::EinSum,
    },
};

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
        is_indice_packed: bool,
    ) -> TractResult<Tensor> {
        if is_indice_packed {
            unimplemented!("unpacking indices not implemented yet !");
        }
        let mut indices = indices.clone();
        let [num_codebooks, num_centroids, vector_len] = *centroids.shape() else {
            unimplemented!("unexected centroid shape ?")
        };

        let [_, _, group_size] = *indices.shape() else {
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
        let indices = indices.into_tensor();
        let centroids = centroids.into_tensor();
        let outlier_indices = outlier_indices.into_tensor();
        let outlier_centroids = outlier_centroids.into_tensor();
        let perm = perm.into_tensor();
        let weight_scale = weight_scale.into_tensor();
        let weight_bias = weight_bias.into_tensor();
        let bias = bias.into_tensor();

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
        assert_eq!(input.rank(), 2);
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

        let is_indice_packed = false; // TODO: apply
        let mut qweight =
            self.eval_extract_from_vector_quant(centroids, indices, is_indice_packed)?;
        if enable_outlier {
            // same as centroids to qweights except for outlier
            let outlier_qweight = self.eval_extract_from_vector_quant(
                outlier_centroids,
                outlier_indices,
                is_indice_packed,
            )?;
            qweight = Tensor::stack_tensors(1, &[outlier_qweight, qweight])?;
        }

        let enable_perm = perm.len() > 1;
        if enable_perm {
            let axis = 0;
            let dim = perm.shape()[0];
            let top_k = Topk { axis, largest: false, fallback_k: dim.into() };
            let invert_perm = top_k
                .eval(tvec!(
                    if is_indice_packed {
                        unimplemented!("permutation not implemented yet with indice packed");
                        // self.perm.to(torch.uint16).to(torch.int64)
                    } else {
                        perm.into_tvalue()
                    },
                    tensor0(dim as u16).into()
                ))?
                .remove(0);
            // TODO: manage case with quant dim == 'in' ?
            // if self.vector_quant_dim == "in":
            //     assert True, "Not implemented"
            //     qweight = qweight[invert_perm, :]

            let perm_gather_axis = 1;
            let gather_perm = Gather { axis: perm_gather_axis };
            qweight = gather_perm
                .eval(tvec!(qweight.into(), invert_perm))?
                .pop()
                .context("apply gather to permutation")
                .unwrap()
                .into_tensor();
        }

        if enable_norm {
            qweight = (qweight.into_array::<f32>()? * weight_scale.into_array::<f32>()?
                + weight_bias.into_array::<f32>()?)
            .into_tensor();
        }
        // call matmul now with qweight

        let einsum_op = EinSum::new("ik,kj->".parse()?, f32::datum_type());
        einsum_op.eval(tvec!(input, qweight.permute_axes(&[1, 0])?.into_tvalue()))
    }
}

impl TypedOp for VPTQGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut tfact = inputs[0].without_value();
        tfact.shape.set(1, self.out_features.into());
        Ok(tvec!(tfact))
    }

    as_op!();
}
