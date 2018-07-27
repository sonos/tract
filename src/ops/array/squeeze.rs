use ops::prelude::*;
use analyser::interface::*;

pub fn squeeze(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let mut dims = pb.get_attr_opt_list_int("squeeze_dims")?;
    if let Some(ref mut dims) = dims {
        dims.sort();
        dims.reverse();
    }
    let t = pb.get_attr_datatype("T")?;
    Ok(boxed_new!(Squeeze(t)(dims)))
}

#[derive(Debug, Clone, new)]
pub struct Squeeze<T: Datum> {
    dims: Option<Vec<isize>>,
    t: PhantomData<T>,
}

impl<T: Datum> Squeeze<T> {
    /// Removes the dimensions of size 1 from the given shape vector.
    fn squeeze_shape(&self, mut shape: Vec<usize>, stream_dim: Option<usize>) -> Result<Vec<usize>> {
        if let Some(ref dims) = self.dims {
            for d in dims {
                if *d >= 0 {
                    shape.remove(*d as usize);
                } else {
                    Err(format!("unimplemented Squeeze with negative parameter"))?
                }
            }
            Ok(shape)
        } else {
            Ok(shape.into_iter().enumerate().filter(|&(ix, d)| stream_dim == Some(ix) || d != 1).map(|(_,d)| d).collect())
        }
    }
}

impl<T: Datum> Op for Squeeze<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let shape = self.squeeze_shape(data.shape().to_vec(), None)?;
        Ok(vec![Tensor::from(data.clone().into_shape(shape)?).into()])
    }
/*
    /// Infers properties about the input and output tensors.
    /// TODO(liautaud): This is ugly, rewrite using the solver.
    fn infer(
        &self,
        inputs: Vec<TensorFact>,
        mut outputs: Vec<TensorFact>,
    ) -> Result<(Vec<TensorFact>, Vec<TensorFact>)> {
        if inputs.len() != 1 {
            bail!("Squeeze operation only supports one input.");
        }

        if outputs.len() != 1 {
            bail!("Squeeze operation only supports one output.");
        }

        let output: Result<_> = {
            // We can't say anything interesting if there are unknown dimensions,
            // because they could turn out to be Only(1), and so Squeeze would
            // have to remove them.
            let shape = match inputs[0].shape.concretize() {
                Some(shape) => self.squeeze_shape(shape, None)?.iter().cloned().collect(),
                None => shapefact![..],
            };

            let output = TensorFact {
                datatype: inputs[0].datatype,
                shape,
                value: valuefact!(_),
            };

            Ok(Some(vec![output]))
        };

        let output = match output? {
            Some(v) => unify(&outputs[0], &v[0])?,
            None => outputs.remove(0),
        };

        let input = unify(&inputs[0], &TensorFact {
            datatype: output.datatype,
            shape: shapefact![..],
            value: valuefact!(_),
        })?;

        Ok((vec![input], vec![output]))
    }
*/

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        let mut attrs = hashmap!{ "T" => Attr::DataType(T::datatype()) };
        if let Some(dim) = self.dims.as_ref() {
            attrs.insert("dims", Attr::IsizeVec(dim.clone()));
        }
        attrs
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: Vec<(Option<usize>, Option<TensorView>)>,
        _buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<Vec<TensorView>>> {
        let input = args_1!(inputs);
        if let (Some(stream), Some(chunk)) = input {
            let chunk = T::tensor_into_array(chunk.into_tensor())?;
            let shape = self.squeeze_shape(chunk.shape().to_vec(), Some(stream))?;
            Ok(Some(vec!( T::array_into_tensor(chunk.into_shape(shape)?).into() )))
        } else {
            Ok(None)
        }
    }
}

impl<T:Datum> InferenceRulesOp for Squeeze<T> {
    fn rules<'r, 'p: 'r>(&self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datatype, &outputs[0].datatype)
            .equals(&inputs[0].datatype, T::datatype());
        ;
        // solver.given(&inputs[0].shape, |solver, shape: Vec<usize>| {
        //     unimplemented!("rules for Squeeze");
        //     // let output_shape = self.squeeze_shape(shape.to_vec(), None);
        //     // FIXME
        // });
    }
}

