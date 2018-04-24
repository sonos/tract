use {Matrix, Result};
use super::{Input, Op, OpRegister};

pub mod local_patch;
pub mod conv2d;
pub mod pools;
pub mod space_to_batch;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("AvgPool", pools::pool::<pools::AvgPooler>);
    reg.insert("Conv2D", conv2d::conv2d);
    reg.insert("MaxPool", pools::pool::<pools::MaxPooler>);
    reg.insert("Relu", Relu::build);
    reg.insert("Softmax", Softmax::build);
    reg.insert("SpaceToBatchND", space_to_batch::space_to_batch_nd);
    reg.insert("BatchToSpaceND", space_to_batch::batch_to_space_nd);
}

element_map!(Relu, |x| if x < 0.0 { 0.0 } else { x });

#[derive(Debug)]
pub struct Softmax {}

impl Softmax {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Softmax {}))
    }
}

impl Op for Softmax {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let m_input = args_1!(inputs);
        let mut input = m_input
            .into_matrix()
            .take_f32s()
            .ok_or("Expect input #0 to be f32")?;
        input.map_inplace(|a| *a = a.exp());
        let norm: f32 = input.iter().sum();
        input.map_inplace(|a| *a = *a / norm);
        let result = Matrix::from(input);
        Ok(vec![result.into()])
    }
}

pub fn arr4<A, V, U, T>(xs: &[V]) -> ::ndarray::Array4<A>
where
    V: ::ndarray::FixedInitializer<Elem = U> + Clone,
    U: ::ndarray::FixedInitializer<Elem = T> + Clone,
    T: ::ndarray::FixedInitializer<Elem = A> + Clone,
    A: Clone,
{
    use ndarray::*;
    let mut xs = xs.to_vec();
    let dim = Ix4(xs.len(), V::len(), U::len(), T::len());
    let ptr = xs.as_mut_ptr();
    let len = xs.len();
    let cap = xs.capacity();
    let expand_len = len * V::len() * U::len() * T::len();
    ::std::mem::forget(xs);
    unsafe {
        let v = if ::std::mem::size_of::<A>() == 0 {
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
        } else if V::len() == 0 || U::len() == 0 || T::len() == 0 {
            Vec::new()
        } else {
            let expand_cap = cap * V::len() * U::len() * T::len();
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
        };
        ArrayBase::from_shape_vec_unchecked(dim, v)
    }
}
