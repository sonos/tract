use tract_data::itertools::Itertools;
use tract_ndarray::Array1;

use crate::{
    internal::*,
    ops::{
        array::{Gather, GatherElements, Topk},
        einsum::EinSum,
        math::shift_left,
    },
};
use tract_linalg::{mmm::{FusedSpec, Packing}, ops};

#[derive(Debug, Clone)]
pub struct VPTQGemm {
    pub vector_len: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub is_indice_packed: bool,
    pub group_size: usize,
    pub outlier_size: usize
}

impl Op for VPTQGemm {
    fn name(&self) -> Cow<str> {
        "VPTQGemm".into()
    }

    op_as_typed_op!();
}
fn shift_right_zero_and_1(input: TValue, shift_value: TValue) -> TractResult<TValue> {
    let input = input.to_array_view::<i32>()?;
    let shift_value = shift_value.to_array_view::<i32>()?;
    let out_shape = crate::broadcast::multi_broadcast(&[input.shape(), shift_value.shape()])?;
    let mut out = Tensor::zero_dt(DatumType::I32, &out_shape)?;
    crate::ndarray::Zip::from(out.to_array_view_mut::<i32>()?)
        .and_broadcast(input)
        .and_broadcast(shift_value)
        .for_each(|c, a, b| *c = a.checked_shr(*b as u32).unwrap_or(0i32) & 1i32);
    Ok(out.into_tvalue())
}

impl VPTQGemm {
    /// decompression of indexes
    fn eval_unpack_index_tensor(
        &self,
        pack_tensor: Tensor,
        index_bits: usize,
        num_elements: usize,
    ) -> TractResult<Tensor> {
        let wf = tensor1(&(0..32i32).collect_vec()).into_shape(&[1, 1, 1, 32])?;

        let pack_tensor_shape = pack_tensor.shape().to_vec();


        let mut pre_shift_pack_tensor_shape = pack_tensor_shape.clone();
        pre_shift_pack_tensor_shape.push(1);

        let mut out = shift_right_zero_and_1(
            pack_tensor.clone().into_shape(&pre_shift_pack_tensor_shape)?.into(),
            wf.into(),
        )?;

        let mut post_shift_pack_tensor_shape = pack_tensor_shape.clone();
        let pval = post_shift_pack_tensor_shape.pop().unwrap();
        post_shift_pack_tensor_shape.push(32 * pval);
        out = out.into_tensor().clone().into_shape(&post_shift_pack_tensor_shape)?.into_tvalue();

        let pad_size = (pack_tensor_shape.last().unwrap_or(&0) * 32) % (index_bits * num_elements);
        if pad_size > 0 {
            let end = out.shape()[out.rank() - 1] - pad_size;
            out = out.slice(out.rank() - 1, 0, end)?.into();
        }

        let mut post_pad_pack_tensor_shape = pack_tensor_shape.clone();
        post_pad_pack_tensor_shape.pop();
        let auto = out.shape().last().unwrap() / index_bits;
        post_pad_pack_tensor_shape.push(auto);
        post_pad_pack_tensor_shape.push(index_bits);
        out = out.into_tensor().into_shape(&post_pad_pack_tensor_shape)?.into();

        let wf1 = Tensor::from(
            Array1::from_iter(0..(index_bits as i32)).to_shape([1, 1, 1, index_bits])?.into_owned(),
        );

        out = shift_left().eval(tvec!(out, wf1.into()))?.pop().unwrap();

        let axis = out.rank() - 1;
        out = out
            .into_tensor()
            .into_array::<i32>()?
            .sum_axis(tract_ndarray::Axis(axis))
            .into_tvalue();

        let unpack_indice = out.cast_to_dt(DatumType::I32)?;

        let mut indices =
            unsafe { Tensor::uninitialized_dt(DatumType::I32, unpack_indice.shape())? };

        crate::ndarray::Zip::from(&mut indices.to_array_view_mut::<i32>()?)
            .and_broadcast(unpack_indice.to_array_view::<i32>()?)
            .for_each(|indice, upack_indice| *indice = upack_indice & ((1 << index_bits) - 1));
        indices = indices.slice(2, 0, num_elements)?;

        Ok(indices)
    }

    fn eval_extract_from_vector_quant(
        &self,
        centroids: Tensor,
        indices: Tensor,
        group_size: usize
    ) -> TractResult<Tensor> {
        /// TODO: instead use ndarray for indices to use views transform (--)
        let mut indices = indices.clone();
        let [num_codebooks, num_centroids, vector_len] = *centroids.shape() else {
            unimplemented!("unexected centroid shape ?")
        };

        if self.is_indice_packed {
            // unimplemented!("unpacking indices not implemented yet !");
            let index_bits = (num_centroids as f32).log2().ceil() as usize;
            indices = self.eval_unpack_index_tensor(indices, index_bits, group_size)?;
        }

        let mut vsh = indices.shape().to_vec();
        indices.insert_axis(3)?;
        vsh.push(vector_len);
        indices = indices.broadcast_to_shape(&vsh)?; // NOTE: costly in tract (applied in memory but not in ndarray)
        let intermediate_volume = indices.shape()[1..3].iter().product();
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
            .permute_axes(&[0, 1, 3, 2])? // NOTE: costly in tract (applied in memory)
            .into_shape(&[num_codebooks, remain * vector_len, group_size])?
            .permute_axes(&[1, 0, 2])?// NOTE: costly in tract (applied in memory)
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
        assert_eq!(indices.datum_type(), DatumType::I32);
        assert_eq!(centroids.rank(), 3);
        assert!(centroids.datum_type().is_float());

        let enable_outlier = outlier_indices.len() > 0;
        if enable_outlier {
            assert_eq!(outlier_indices.rank(), 3);
            assert_eq!(outlier_indices.datum_type(), DatumType::I32);
            assert_eq!(outlier_centroids.rank(), 3);
            assert!(outlier_centroids.datum_type().is_float());
        }

        let mut qweight = self.eval_extract_from_vector_quant(centroids, indices, self.group_size)?;
        if enable_outlier {
            // same as centroids to qweights except for outlier
            let outlier_qweight =
                self.eval_extract_from_vector_quant(outlier_centroids, outlier_indices, self.outlier_size)?;
            qweight = Tensor::stack_tensors(1, &[outlier_qweight, qweight])?;
        }

        let enable_perm = perm.len() > 1;
        if enable_perm {
            let axis = 0;
            let dim = perm.shape()[0];
            let top_k = Topk { axis, largest: false, fallback_k: dim.into() };
            let invert_perm =
                top_k.eval(tvec!(perm.into_tvalue(), tensor0(dim as u16).into()))?.remove(0);
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
        // // call matmul now with qweight
        // let einsum_op = EinSum::new("ik,kj->ij".parse()?, f32::datum_type());
        // // einsum -> matmul imperatif
        // einsum_op.eval(tvec!(input, qweight.permute_axes(&[1, 0])?.into_tvalue()))
        //
        qweight = qweight.permute_axes(&[1, 0])?;
        let op = ops();
        let &[m, k] = input.shape() else { bail!("unexpected rank {:?}", input.rank())};
        let &n = qweight.shape().last().unwrap();

        let mmm = op.mmm(DatumType::F32, Some(m), Some(k), Some(n)).unwrap();

        let (pack_a, pack_b) = &mmm.packings()[0];
        let cstore = unsafe {
            mmm.c_view(0, 1)
        };


        let a = pack_a.prepare_tensor(&input, 1, 0)?;
        let b = pack_b.prepare_tensor(&qweight, 0, 1)?;
        unsafe {
            let mut out =
                Tensor::uninitialized::<f32>(&[m, n])?;
        let non_linear = &[FusedSpec::AddMatMul {
            a: tract_linalg::mmm::AsInputValue::Owned(a), b: tract_linalg::mmm::AsInputValue::Owned(b), packing: 0
        }, FusedSpec::Store(cstore.wrap(&out.view()))];
            mmm.run(m, n, non_linear);

        Ok(tvec!(out.into()))
        }
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
