use crate::prelude::TVec;
use itertools::Itertools;
use std::fmt;

#[derive(Clone, Default)]
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

impl fmt::Debug for Invariants {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if self.element_wise {
            write!(fmt, "Element wise")?;
        } else if self.axes.len() > 0 {
            write!(fmt, "Axes tracking: {}", self.axes.iter().map(|axis| format!("{:?}", axis)).join(", "))?;
        } else {
            write!(fmt, "No invariants")?;
        }
        Ok(())
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
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AxisInfo {
    pub inputs: TVec<Option<usize>>,
    pub outputs: TVec<Option<usize>>,
    pub period: usize,
    pub disposable: bool,
}

impl fmt::Debug for AxisInfo {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}->{}",
               self.inputs.iter().map(|i| i.map(|a| a.to_string()).unwrap_or("_".to_string())).join(","),
               self.outputs.iter().map(|i| i.map(|a| a.to_string()).unwrap_or("_".to_string())).join(","))?;
        if !self.disposable {
            write!(fmt, " not disposable")?;
        }
        if self.period != 1 {
            write!(fmt, " period: {}", self.period)?;
        }
        Ok(())
    }
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
