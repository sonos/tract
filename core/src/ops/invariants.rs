use crate::prelude::TVec;

#[derive(Debug, Clone, Default)]
pub struct Invariants {
    element_wise: bool,
    axes: TVec<AxisInfo>,
}

impl Invariants {
    pub fn none() -> Invariants {
        Invariants { element_wise: false, axes: tvec!() }
    }

    pub fn new_element_wise() -> Invariants {
        Invariants { element_wise: true, axes: tvec!() }
    }

    pub fn element_wise(&self) -> bool {
        self.element_wise
    }
}

impl From<TVec<AxisInfo>> for Invariants {
    fn from(axes: TVec<AxisInfo>) -> Invariants {
        Invariants { element_wise: false, axes }
    }
}

impl std::iter::FromIterator<AxisInfo> for Invariants {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = AxisInfo>,
    {
        Invariants { element_wise: false, axes: iter.into_iter().collect() }
    }
}

/// Translation invariance property.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AxisInfo {
    pub inputs: TVec<Option<usize>>,
    pub outputs: TVec<Option<usize>>,
    pub period: usize,
    pub disposable: bool,
}

impl AxisInfo {
    pub fn simple(axis: usize) -> AxisInfo {
        AxisInfo {
            inputs: tvec!(Some(axis)),
            outputs: tvec!(Some(axis)),
            period: 1,
            disposable: true,
        }
    }

    pub fn with_period(self, period: usize) -> AxisInfo {
        AxisInfo { period, ..self }
    }

    pub fn disposable(self, disposable: bool) -> AxisInfo {
        AxisInfo { disposable, ..self }
    }
}

impl Invariants {
    pub fn unary_track_axis_up(&self, axis: usize, only_disposable: bool) -> Option<usize> {
        if self.element_wise {
            Some(axis)
        } else {
            self.axes
                .iter()
                .find(|connection| {
                    connection.outputs.get(0) == Some(&Some(axis)) && connection.period == 1
                })
                .filter(|conn| conn.disposable || !only_disposable)
                .and_then(|connection| connection.inputs.get(0))
                .and_then(|d| *d)
        }
    }

    pub fn unary_track_axis_down(&self, axis: usize, only_disposable: bool) -> Option<usize> {
        if self.element_wise {
            Some(axis)
        } else {
            self.axes
                .iter()
                .find(|connection| {
                    connection.inputs.get(0) == Some(&Some(axis)) && connection.period == 1
                })
                .filter(|conn| conn.disposable || !only_disposable)
                .and_then(|connection| connection.outputs.get(0))
                .and_then(|d| *d)
        }
    }
}
