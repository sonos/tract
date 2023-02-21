use proptest::arbitrary::Arbitrary;
use proptest::prelude::*;
use proptest::strategy::{BoxedStrategy, Strategy};
use tract_data::internal::*;
use tract_linalg::frame::mmm::FusedSpec;
use tract_linalg::frame::mmm::{VirtualInput, VirtualInputSpec};
use tract_linalg::frame::PackingWriter;
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

    pub fn tract(&self) -> Tensor {
        let (m, k, n, h, w) = mknhw(self.filters.shape(), self.input.shape());
        let output_shape = [m, h, w];
        let internal_output_shape = [m, h * w];
        let mmm = tract_linalg::ops().mmm(F32, F32, F32, Some(m), Some(k), Some(n)).unwrap();
        let output = Tensor::zero::<f32>(&internal_output_shape).unwrap();
        let mut packed_filter =
            Tensor::zero_aligned::<f32>(&[mmm.a_pack().len(k, m)], mmm.a_pack().alignment())
                .unwrap();
        let reshaped_filters = self.filters.clone().into_shape(&[k, m]).unwrap();
        unsafe {
            mmm.a_pack().pack(packed_filter.view_mut(), reshaped_filters.view(), 0, 1);
            let a_store = mmm.a_packed(F32.size_of(), k).wrap(&packed_filter.view());
            let im2col: Box<dyn VirtualInputSpec> = if self.lazy_im2col {
                Box::new(LazyIm2colSpec { full_kernel_shape: self.filters.shape().into() })
            } else {
                Box::new(EagerIm2colSpec { full_kernel_shape: self.filters.shape().into() })
            };
            let b_store = mmm.b_virtual_input(im2col, k).wrap(&self.input.view());
            let c_store = mmm.c_view(0, 1).wrap(&output.view());
            mmm.run(
                m,
                n,
                &[FusedSpec::AddMatMul { k, a: a_store, b: b_store }, FusedSpec::Store(c_store)],
            )
            .unwrap()
        }
        output.into_shape(&output_shape).unwrap()
    }

    fn check(&self) {
        let found = self.tract();
        let expected = self.reference();
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
            .prop_flat_map(|(eager_im2col, h, w, i, o, extra_h, extra_w)| {
                let filters = tensor(vec![h, w, i, o]);
                let input = tensor(vec![i, h + extra_h, w + extra_w]);
                (Just(eager_im2col), filters, input)
            })
            .prop_map(|(eager_im2col, filters, input)| ConvProblem {
                lazy_im2col: eager_im2col,
                filters,
                input,
            })
            .boxed()
    }
}

fn tensor(shape: Vec<usize>) -> BoxedStrategy<Tensor> {
    let len = shape.iter().product::<usize>();
    proptest::collection::vec(any::<i8>(), len..=len)
        .prop_map(move |vec| {
            tract_ndarray::ArrayD::from_shape_vec(shape.clone(), vec)
                .unwrap()
                .into_tensor()
                .cast_to_dt(F32)
                .unwrap()
                .into_owned()
        })
        .boxed()
}

#[derive(Clone, Debug, Hash)]
struct EagerIm2colSpec {
    full_kernel_shape: TVec<usize>,
}

impl VirtualInputSpec for EagerIm2colSpec {
    fn wrap(&self, input: &TensorView) -> Box<dyn VirtualInput> {
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
        Box::new(EagerIm2col { im2col: im2col.into_tensor() })
    }
}

#[derive(Clone, Debug)]
struct EagerIm2col {
    im2col: Tensor,
}

impl VirtualInput for EagerIm2col {
    fn input(
        &self,
        packer: &tract_linalg::frame::Packer,
        packed: *mut u8,
        k_range: std::ops::Range<usize>,
        mn_range: std::ops::Range<usize>,
    ) {
        let mn = self.im2col.shape()[1];
        unsafe {
            packer.pack_t::<f32>(
                packed as _,
                self.im2col.as_ptr().unwrap(),
                mn,
                mn as isize,
                1,
                k_range,
                mn_range,
            );
        }
    }
}

#[derive(Clone, Debug, Hash)]
struct LazyIm2colSpec {
    full_kernel_shape: TVec<usize>,
}

impl VirtualInputSpec for LazyIm2colSpec {
    fn wrap(&self, input: &TensorView) -> Box<dyn VirtualInput> {
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
        unsafe { Box::new(LazyIm2col { image: input.as_ptr_unchecked(), k_offsets, n_offsets }) }
    }
}

#[derive(Clone, Debug)]
struct LazyIm2col {
    image: *const f32,
    n_offsets: Vec<isize>,
    k_offsets: Vec<isize>,
}
unsafe impl Send for LazyIm2col {}
unsafe impl Sync for LazyIm2col {}

impl VirtualInput for LazyIm2col {
    fn input(
        &self,
        packer: &tract_linalg::frame::Packer,
        packed: *mut u8,
        k_range: std::ops::Range<usize>,
        mn_range: std::ops::Range<usize>,
    ) {
        let mn_end = mn_range.end.min(self.n_offsets.len());
        let n_range = mn_range.start..mn_end;
        unsafe {
            let mut writer = packer.write_with_k_outer(packed as _, k_range.len(), n_range.len());
            for k in k_range.start..k_range.end {
                for n in n_range.start..n_range.end {
                    writer.write(
                        *self.image.offset(
                            self.n_offsets.get_unchecked(n) + self.k_offsets.get_unchecked(k),
                        ),
                    )
                }
            }
        }
    }
}
