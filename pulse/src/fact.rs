use crate::internal::*;

lazy_static::lazy_static! {
    static ref S: Symbol = Symbol::new('S');
}

pub fn stream_symbol() -> Symbol {
    *S
}

pub fn stream_dim() -> TDim {
    (*S).into()
}

pub trait StreamFact {
    fn stream_info(&self) -> Option<(usize, &TDim)>;
}

impl StreamFact for ShapeFact {
    fn stream_info(&self) -> Option<(usize, &TDim)> {
        let streaming_dims: TVec<(usize, &TDim)> = (&**self)
            .iter()
            .enumerate()
            .filter(|(_ix, d)| d.symbols().contains(&stream_symbol()))
            .collect();
        if streaming_dims.len() != 1 {
            None
        } else {
            Some(streaming_dims[0])
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PulsedFact {
    pub datum_type: DatumType,
    pub shape: ShapeFact,
    pub axis: usize,
    pub dim: TDim,
    pub delay: usize,
}

impl_dyn_hash!(PulsedFact);

impl PulsedFact {
    pub fn from_tensor_fact_pulse(tf: &TypedFact, pulse: usize) -> TractResult<PulsedFact> {
        let datum_type = tf.datum_type;
        let (axis, len) = tf
            .shape
            .stream_info()
            .ok_or_else(|| format_err!("Can not pulse a tensor with no streaming dim"))?;
        let mut shape: TVec<TDim> = tf.shape.iter().collect();
        shape[axis] = pulse.into();
        Ok(PulsedFact { datum_type, shape: shape.into(), axis, dim: len.clone(), delay: 0 })
    }

    pub fn pulse(&self) -> usize {
        self.shape[self.axis].to_usize().expect("Pulse should be an integer. This is a tract bug.")
    }

    pub fn to_pulse_fact(&self) -> TypedFact {
        self.datum_type.fact(self.shape.clone())
    }

    pub fn streaming_shape(&self) -> Vec<TDim> {
        self.shape
            .iter()
            .enumerate()
            .map(|(ix, d)| if ix == self.axis { self.dim.clone() } else { d })
            .collect()
    }

    pub fn to_streaming_fact(&self) -> TypedFact {
        let mut info = self.to_pulse_fact();
        info.shape.set(self.axis, self.dim.clone());
        info
    }
}

impl fmt::Debug for PulsedFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use tract_itertools::Itertools;
        write!(
            fmt,
            "{},{:?} [pulse axis:{} ∂:{} full dim:{}]",
            self.shape.iter().join(","),
            self.datum_type,
            self.axis,
            self.delay,
            self.dim
        )
    }
}

impl Fact for PulsedFact {
    fn to_typed_fact(&self) -> TractResult<Cow<TypedFact>> {
        Ok(Cow::Owned(self.into()))
    }

    fn same_as(&self, other: &dyn Fact) -> bool {
        if let Some(other) = other.downcast_ref::<PulsedFact>() {
            other == self
        } else {
            false
        }
    }

    fn compatible_with(&self, other: &dyn Fact) -> bool {
        self.same_as(other)
    }

    fn datum_type(&self) -> Option<DatumType> {
        Some(self.datum_type)
    }
}

impl From<PulsedFact> for TypedFact {
    fn from(fact: PulsedFact) -> TypedFact {
        fact.datum_type.fact(fact.shape)
    }
}

impl<'a> From<&'a PulsedFact> for TypedFact {
    fn from(fact: &'a PulsedFact) -> TypedFact {
        fact.datum_type.fact(fact.shape.clone())
    }
}
