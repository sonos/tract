use std::alloc::Layout;
use std::fmt::Display;

use proptest::arbitrary::Arbitrary;
use proptest::prelude::*;
use proptest::strategy::{BoxedStrategy, Strategy};
use tract_data::internal::*;
use tract_linalg::frame::mmm::FusedSpec;
// use tract_linalg::frame::mmm::{VirtualInput, VirtualInputSpec};
use tract_linalg::frame::{Packer, PackingWriter};
use tract_linalg::mmm::MMMInputValue;
use DatumType::F32;

proptest::proptest! {
    #[test]
    fn prop(pb in any::<ConvProblem>()) {
        pb.check()
    }
}

#[test]
fn test1() {
    ConvProblem {
        lazy_im2col: false,
        input: tensor3(&[[[1f32]]]),
        filters: tensor4(&[[[[-1f32]]]]),
    }
    .check()
}

#[test]
fn test_axes_0() {
    // CHW HWIO CHW
    // 121 1112 221
    ConvProblem {
        lazy_im2col: false,
        input: tensor3(&[[[0f32], [-1.0]]]),
        filters: tensor4(&[[[[0f32, -1f32]]]]),
    }
    .check()
}

#[test]
fn test_axes_1() {
    ConvProblem {
        lazy_im2col: false,
        input: tensor3(&[[[0f32, 1.]]]),
        filters: tensor4(&[[[[1f32]]]]),
    }
    .check()
}

#[test]
fn test_lazy_0() {
    ConvProblem { lazy_im2col: true, input: tensor3(&[[[1f32]]]), filters: tensor4(&[[[[1f32]]]]) }
        .check()
}

#[test]
fn test_lazy_1() {
    ConvProblem {
        lazy_im2col: true,
        input: tensor3(&[[[0f32], [0.], [0.]]]),
        filters: tensor4(&[[[[0f32]]]]),
    }
    .check()
}

#[test]
fn test_lazy_2() {
    ConvProblem {
        lazy_im2col: true,
        input: tensor3(&[[[0f32, 0.], [0., 1.]]]),
        filters: tensor4(&[[[[0f32]], [[1.]]]]),
    }
    .check()
}

#[test]
fn test_lazy_3() {
    // CHW HWIO CHW
    // 212 1221 111
    // im2col: k=4, n=1, k <- kh, kw, c
    // 0 X X X X kh=0, kw=0, c=0
    // 1 X X X X kh=0, kw=0, c=1
    // 0 X X X X kh=0, kw=1, c=0
    // 0 X X X X kh=0, kw=1, c=1
    ConvProblem {
        lazy_im2col: true,
        input: tensor3(&[[[0f32, 0.]], [[1., 0.]]]),
        filters: tensor4(&[[[[0f32], [0.]], [[1.], [0.]]]]),
    }
    .check()
}

#[test]
fn test_eager_asan_0() {
    ConvProblem {
        lazy_im2col: false,
        input: tensor(vec![3, 3, 5]),
        filters: tensor(vec![3, 3, 3, 1]),
    }
    .check()
}

// 2D valid, no group, no dil, no stride, HWIO, CHW
#[derive(Clone, Debug)]
pub struct ConvProblem {
    pub lazy_im2col: bool,
    pub input: Tensor,
    pub filters: Tensor,
}

fn mknhw(filters: &[usize], input: &[usize]) -> (usize, usize, usize, usize, usize) {
    let m = filters[3];
    let k = filters[0..3].iter().product::<usize>();
    let h = input[1] - filters[0] + 1;
    let w = input[2] - filters[1] + 1;
    let n = h * w;
    (m, k, n, h, w)
}

impl ConvProblem {
    fn reference(&self) -> Tensor {
        let (m, _, _, h, w) = mknhw(self.filters.shape(), self.input.shape());
        let output_shape = [m, h, w];
        let mut output = Tensor::zero::<f32>(&output_shape).unwrap();
        let mut output_view = output.to_array_view_mut::<f32>().unwrap();
        let input_view = self.input.to_array_view::<f32>().unwrap();
        let filters_view = self.filters.to_array_view::<f32>().unwrap();
        for geo_out in tract_ndarray::indices(&output_shape[1..]) {
            for ker_geo in tract_ndarray::indices(&self.filters.shape()[0..2]) {
                for ci in 0..self.filters.shape()[2] {
                    for co in 0..self.filters.shape()[3] {
                        let output_coord = [co, geo_out[0], geo_out[1]];
                        let input_coord = [ci, geo_out[0] + ker_geo[0], geo_out[1] + ker_geo[1]];
                        let ker_coord = [ker_geo[0], ker_geo[1], ci, co];
                        output_view[output_coord] +=
                            filters_view[ker_coord] * input_view[input_coord];
                    }
                }
            }
        }
        output
    }

    pub fn tract(&self) -> TractResult<Tensor> {
        let (m, k, n, h, w) = mknhw(self.filters.shape(), self.input.shape());
        let output_shape = [m, h, w];
        let internal_output_shape = [m, h * w];
        let mmm = tract_linalg::ops().mmm(F32, F32, F32, Some(m), Some(k), Some(n)).unwrap();
        let output = Tensor::zero::<f32>(&internal_output_shape)?;
        let reshaped_filters = self.filters.clone().into_shape(&[k, m])?;
        let (a_pack, b_pack) = mmm.packings()[0];
        let a = a_pack.prepare_tensor(&reshaped_filters, 0, 1)?;
        unsafe {
            let im2col: Box<dyn MMMInputValue> = if self.lazy_im2col {
                LazyIm2colSpec {
                    full_kernel_shape: self.filters.shape().into(),
                    packer: b_pack.downcast_ref::<Packer>().unwrap().clone(),
                }
                .wrap(&self.input.view())
            } else {
                EagerIm2colSpec {
                    full_kernel_shape: self.filters.shape().into(),
                    packer: b_pack.downcast_ref::<Packer>().unwrap().clone(),
                }
                .wrap(&self.input.view())
            };
            let c_store = mmm.c_view(0, 1).wrap(&output.view());
            mmm.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul { a: &*a, b: &*im2col, packing: 0 },
                    FusedSpec::Store(c_store),
                ],
            )
            .unwrap()
        }
        output.into_shape(&output_shape)
    }

    fn check(&self) {
        let expected = self.reference();
        let found = self.tract().unwrap();
        if found.close_enough(&expected, true).is_err() {
            println!("found: ");
            println!("{:?}", found.to_array_view::<f32>().unwrap());
            println!("expected: ");
            println!("{:?}", expected.to_array_view::<f32>().unwrap());
        }
        found.close_enough(&expected, true).unwrap()
    }
}

impl Arbitrary for ConvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (any::<bool>(), 1..4usize, 1..4usize, 1..4usize, 1..4usize, 0..3usize, 0..3usize)
            .prop_map(|(eager_im2col, h, w, i, o, extra_h, extra_w)| {
                let filters = tensor(vec![h, w, i, o]);
                let input = tensor(vec![i, h + extra_h, w + extra_w]);
                ConvProblem { lazy_im2col: eager_im2col, filters, input }
            })
            .boxed()
    }
}

fn tensor(shape: Vec<usize>) -> Tensor {
    let mut tensor = Tensor::zero::<f32>(&shape).unwrap();
    tensor.as_slice_mut::<f32>().unwrap().iter_mut().enumerate().for_each(|(ix, x)| *x = ix as f32);
    tensor
}

#[derive(Clone, Debug, Hash)]
struct EagerIm2colSpec {
    packer: Packer,
    full_kernel_shape: TVec<usize>,
}

impl EagerIm2colSpec {
    fn wrap(&self, input: &TensorView) -> Box<dyn MMMInputValue> {
        let (_, k, n, h, w) = mknhw(&self.full_kernel_shape, input.shape());
        // let input = input.to_array_view::<f32>().unwrap();
        let ci = input.shape()[0];
        let kh = self.full_kernel_shape[0];
        let kw = self.full_kernel_shape[1];
        let im2col = tract_ndarray::Array5::<f32>::from_shape_fn(
            [kh, kw, ci, h, w],
            |(kh, kw, ci, h, w)| *input.at([ci, h + kh, w + kw]).unwrap(),
        )
        .into_shape([k, n])
        .unwrap();
        Box::new(EagerIm2col { im2col: im2col.into_tensor(), packer: self.packer.clone(), k })
    }
}

#[derive(Clone, Debug, Hash)]
struct EagerIm2col {
    packer: Packer,
    im2col: Tensor,
    k: usize,
}

impl Display for EagerIm2col {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "eager")
    }
}

impl MMMInputValue for EagerIm2col {
    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        Some(
            Layout::from_size_align(
                self.packer.single_panel_len(self.k) * f32::datum_type().size_of(),
                self.packer.alignment(),
            )
            .unwrap(),
        )
    }

    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        let buffer = buffer.unwrap();
        let mn = self.im2col.shape()[1];
        unsafe {
            self.packer.pack_t::<f32>(
                buffer as _,
                self.im2col.as_ptr().unwrap(),
                mn,
                mn as isize,
                1,
                0..self.k,
                (i * self.packer.r)..((i + 1) * self.packer.r),
            );
        }
        Ok(buffer)
    }

    fn k(&self) -> usize {
        self.k
    }

    fn mn(&self) -> usize {
        self.im2col.shape()[1]
    }

    fn r(&self) -> usize {
        self.packer.r
    }
}

#[derive(Clone, Debug, Hash)]
struct LazyIm2colSpec {
    packer: Packer,
    full_kernel_shape: TVec<usize>,
}

impl LazyIm2colSpec {
    fn wrap(&self, input: &TensorView) -> Box<dyn MMMInputValue> {
        let (_, _, _, h, w) = mknhw(&self.full_kernel_shape, input.shape());
        let kh = self.full_kernel_shape[0];
        let kw = self.full_kernel_shape[1];
        let ci = self.full_kernel_shape[2];
        let input_strides = input.strides();
        let k_offsets = (0..kh as isize)
            .flat_map(|kh| {
                (0..kw as isize).flat_map(move |kw| {
                    (0..ci as isize).map(move |ci| {
                        ci * input_strides[0] + kh * input_strides[1] + kw * input_strides[2]
                    })
                })
            })
            .collect();
        let n_offsets = (0..h as isize)
            .flat_map(|h| {
                (0..w as isize).map(move |w| (h * input_strides[1] + w * input_strides[2]))
            })
            .collect();
        unsafe {
            Box::new(LazyIm2col {
                image: input.as_ptr_unchecked(),
                k_offsets,
                n_offsets,
                packer: self.packer.clone(),
            })
        }
    }
}

#[derive(Clone, Debug, Hash)]
struct LazyIm2col {
    packer: Packer,
    image: *const f32,
    n_offsets: Vec<isize>,
    k_offsets: Vec<isize>,
}
unsafe impl Send for LazyIm2col {}
unsafe impl Sync for LazyIm2col {}

impl Display for LazyIm2col {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lazy")
    }
}

impl MMMInputValue for LazyIm2col {
    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        Some(
            Layout::from_size_align(
                self.packer.single_panel_len(self.k_offsets.len() * f32::datum_type().size_of()),
                self.packer.alignment(),
            )
            .unwrap(),
        )
    }

    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        let buffer = buffer.unwrap() as *mut f32;
        let mn_end = ((i + 1) * self.packer.r).min(self.n_offsets.len());
        let n_range = (i * self.packer.r)..mn_end;
        let k = self.k_offsets.len();
        unsafe {
            let mut writer = self.packer.write_with_k_outer(buffer, k, n_range.len());
            for k in 0..k {
                for n in n_range.clone() {
                    writer.write(
                        *self.image.offset(
                            self.n_offsets.get_unchecked(n) + self.k_offsets.get_unchecked(k),
                        ),
                    )
                }
            }
        }
        Ok(buffer as _)
    }

    fn k(&self) -> usize {
        self.k_offsets.len()
    }

    fn mn(&self) -> usize {
        self.n_offsets.len()
    }

    fn r(&self) -> usize {
        self.packer.r
    }
}
