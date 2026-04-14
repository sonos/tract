/// PulseV2: region/increment-based pulsification.
///
/// Instead of tracking a single streaming axis with a fixed pulse size and
/// startup delay, PulseV2 describes per-axis **regions** (what data an op
/// needs or produces at pulse T) and **increments** (what's new compared
/// to pulse T-1).
///
/// Symbols:
///   T — pulse index (0, 1, 2, …)
///   P — pulse size (a concrete positive integer)
///
/// For a given node at pulse T, each axis is either:
///   - Fixed: the full extent is produced every pulse (batch, channel, etc.)
///   - Streaming: a range [start(T,P), end(T,P)) is produced
///
/// The **region** is the full range needed/produced at pulse T.
/// The **increment** is the new data: region(T) \ region(T-1).
/// The **buffer** is the overlap: region(T) ∩ region(T-1) — data that must
/// be retained from the previous pulse.
use crate::internal::*;

/// Per-axis specification at pulse T.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AxisRegion {
    /// This axis is not streaming — full extent every pulse.
    Fixed(TDim),
    /// This axis produces [start, end) at pulse T.
    /// start and end are TDim expressions in symbols T and P.
    Streaming { start: TDim, end: TDim },
}

impl AxisRegion {
    /// Size of this axis at pulse T.
    pub fn size(&self) -> TDim {
        match self {
            AxisRegion::Fixed(d) => d.clone(),
            AxisRegion::Streaming { start, end } => end.clone() - start.clone(),
        }
    }

    /// Is this axis streaming?
    pub fn is_streaming(&self) -> bool {
        matches!(self, AxisRegion::Streaming { .. })
    }
}

/// Region/increment description for a wire at pulse T.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PulseV2Region {
    /// Per-axis region specification.
    pub axes: TVec<AxisRegion>,
}

impl PulseV2Region {
    /// Shape of the tensor at pulse T (one TDim per axis).
    pub fn shape(&self) -> TVec<TDim> {
        self.axes.iter().map(|a| a.size()).collect()
    }

    /// Rank.
    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    /// Indices of streaming axes.
    pub fn streaming_axes(&self) -> TVec<usize> {
        self.axes.iter().enumerate().filter(|(_, a)| a.is_streaming()).map(|(i, _)| i).collect()
    }
}

/// Fact for a wire in a PulseV2 model.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PulseV2Fact {
    pub datum_type: DatumType,
    /// Region at pulse T. None for non-streaming wires (constants, etc.)
    pub region: Option<PulseV2Region>,
    /// The batch (full-stream) shape, with symbolic stream dimensions.
    pub batch_shape: ShapeFact,
}

impl PulseV2Fact {
    /// Create a PulseV2Fact for a streaming input.
    ///
    /// `batch_fact` is the TypedFact from the batch model.
    /// `stream_sym` is the symbol representing total stream length.
    /// `pulse_sym` is the symbol representing pulse size (P).
    /// `pulse_id_sym` is the symbol representing pulse index (T).
    ///
    /// All axes containing `stream_sym` become Streaming with
    /// start/end derived by substituting stream_sym → T*P..(T+1)*P.
    pub fn from_batch_input(
        batch_fact: &TypedFact,
        stream_sym: &Symbol,
        pulse_sym: &Symbol,
        pulse_id_sym: &Symbol,
    ) -> TractResult<Self> {
        let t = TDim::Sym(pulse_id_sym.clone());
        let p = TDim::Sym(pulse_sym.clone());
        let mut axes = TVec::new();

        for dim in batch_fact.shape.iter() {
            if dim.symbols().contains(stream_sym) {
                // This axis depends on the stream symbol.
                // For a simple case like dim=S, region is [T*P, (T+1)*P).
                // For dim=(S+k)/stride, we'd need to transform accordingly.
                // Start simple: only handle dim=S directly.
                if dim == &TDim::Sym(stream_sym.clone()) {
                    axes.push(AxisRegion::Streaming {
                        start: t.clone() * p.clone(),
                        end: (t.clone() + 1) * p.clone(),
                    });
                } else {
                    bail!(
                        "PulseV2: streaming dimension {dim} is not a simple symbol — \
                         complex streaming dims not yet supported"
                    );
                }
            } else {
                axes.push(AxisRegion::Fixed(dim.clone()));
            }
        }

        Ok(PulseV2Fact {
            datum_type: batch_fact.datum_type,
            region: Some(PulseV2Region { axes }),
            batch_shape: batch_fact.shape.clone(),
        })
    }

    /// Shape at pulse T (substituting region sizes).
    pub fn pulse_shape(&self) -> TVec<TDim> {
        match &self.region {
            Some(r) => r.shape(),
            None => self.batch_shape.to_tvec(),
        }
    }

    /// Is this wire streaming?
    pub fn is_streaming(&self) -> bool {
        self.region.is_some()
    }
}

// ── Pulsification ──────────────────────────────────────────────────────────

/// Symbols used during PulseV2 pulsification.
pub struct PulseV2Symbols {
    /// The stream-length symbol in the batch model (e.g. S).
    pub stream: Symbol,
    /// Pulse size symbol (P) — will be set to a concrete value at runtime.
    pub pulse: Symbol,
    /// Pulse index symbol (T) — increments at each step: 0, 1, 2, …
    pub pulse_id: Symbol,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_streaming_input() {
        let scope = SymbolScope::default();
        let s = scope.sym("S");
        let p = scope.sym("P");
        let t = scope.sym("T");

        // Batch shape: [1, S, 16] — a 1D streaming signal with 16 channels
        let batch_fact = DatumType::F32.fact(&[1.to_dim(), s.clone().into(), 16.to_dim()]);
        let fact = PulseV2Fact::from_batch_input(&batch_fact, &s, &p, &t).unwrap();

        assert_eq!(fact.datum_type, DatumType::F32);
        assert!(fact.is_streaming());

        let region = fact.region.as_ref().unwrap();
        assert_eq!(region.rank(), 3);
        assert!(!region.axes[0].is_streaming()); // batch
        assert!(region.axes[1].is_streaming()); // time
        assert!(!region.axes[2].is_streaming()); // channels

        // Shape at pulse T should be [1, P, 16]
        let shape = fact.pulse_shape();
        assert_eq!(shape[0], 1.to_dim());
        assert_eq!(shape[1].clone().simplify(), TDim::Sym(p)); // (T+1)*P - T*P simplifies to P
        assert_eq!(shape[2], 16.to_dim());
    }
}
