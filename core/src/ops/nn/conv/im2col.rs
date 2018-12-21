use tract_linalg::MatMul;

use ndarray::prelude::*;
use ops::prelude::*;

use ops::nn::Patch;

use num::Zero;
use std::ops::Mul;

#[derive(Debug, Clone, new, PartialEq)]
pub(super) struct Im2Col {
    pub patch: Patch,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub packed_b_len: usize,
}

impl Im2Col {
    pub(super) fn im2col<'i, D: Datum + Mul + Zero>(
        &'i self,
        input: &'i ArrayViewD<'i, D>,
        mm: &MatMul<D>,
    ) -> TractResult<Array1<D>> {
        let input_shape = &self.patch.input_shape;
        let mut mega_matrix = unsafe { Array2::<D>::uninitialized((self.k, self.n)) };

        let mut packed = unsafe {
            Array1::<D>::uninitialized(self.packed_b_len * self.group * input_shape.n_dim())
        };
        let visitor = self.patch.wrap(input);
        let ci_per_group = input_shape.c_dim() / self.group;
        for i in 0..input_shape.n_dim() {
            for g in 0..self.group {
                let mut coords = vec![0; input_shape.rank()];
                coords[input_shape.n_axis()] = i;
                for (mut spatial, mut col) in ndarray::indices(&*self.patch.output_spatial_shape)
                    .into_iter()
                    .zip(mega_matrix.axis_iter_mut(Axis(1)))
                {
                    let mut col = col.iter_mut();
                    coords[input_shape.h_axis()..][..input_shape.hw_rank()]
                        .copy_from_slice(spatial.slice());
                    for ci in 0..ci_per_group {
                        coords[input_shape.c_axis()] = ci + g * ci_per_group;
                        for v in visitor.at(&*coords) {
                            *col.next().expect("geometry error in conv") =
                                v.unwrap_or(D::default());
                        }
                    }
                }
                unsafe {
                    mm.pack_b(
                        self.k,
                        self.n,
                        packed
                            .as_mut_ptr()
                            .offset(((i * self.group + g) * self.packed_b_len) as isize),
                        mega_matrix.as_ptr(),
                        mega_matrix.strides()[0],
                        mega_matrix.strides()[1],
                    );
                }
            }
        }
        trace!("im2col: {:?}", packed);
        Ok(packed)
    }
}

impl Op for Im2Col {
    fn name(&self) -> Cow<str> {
        "Im2col".into()
    }

    impl_op_same_as!();
}

impl StatelessOp for Im2Col {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let tensor = match inputs[0].datum_type() {
            DatumType::F32 => self
                .im2col::<f32>(&inputs[0].to_array_view()?, &*tract_linalg::ops().smm)?
                .into(),
            _ => unimplemented!()
        };
        Ok(tvec!(tensor))
    }
}

impl InferenceRulesOp for Im2Col {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(
            &inputs[0].shape,
            ShapeFact::from(&*self.patch.input_shape.shape),
        )?;
        s.equals(
            &outputs[0].shape,
            ShapeFact::from(&[self.packed_b_len * self.group]),
        )?;
        Ok(())
    }
}
