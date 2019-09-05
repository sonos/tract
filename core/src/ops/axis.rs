use crate::prelude::TVec;

#[derive(Debug, Clone)]
pub struct AxesInfo(TVec<AxisInfo>);

impl AxesInfo {
    pub fn none() -> AxesInfo {
        AxesInfo(tvec!())
    }
}

impl From<TVec<AxisInfo>> for AxesInfo {
    fn from(axes: TVec<AxisInfo>) -> AxesInfo {
        AxesInfo(axes)
    }
}

impl std::iter::FromIterator<AxisInfo> for AxesInfo {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = AxisInfo>,
    {
        AxesInfo(iter.into_iter().collect())
    }
}

impl std::ops::Deref for AxesInfo {
    type Target = [AxisInfo];
    fn deref(&self) -> &[AxisInfo] {
        &self.0
    }
}

/// Translation invariance property.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AxisInfo {
    pub inputs: TVec<Option<usize>>,
    pub outputs: TVec<Option<usize>>,
    pub period: usize,
}

impl AxisInfo {
    pub fn simple(axis: usize) -> AxisInfo {
        AxisInfo { inputs: tvec!(Some(axis)), outputs: tvec!(Some(axis)), period: 1 }
    }

    pub fn with_period(self, period: usize) -> AxisInfo {
        AxisInfo { period, ..self }
    }
}

impl AxesInfo {
    pub fn unary_track_axis_up(&self, axis: usize) -> Option<usize> {
        self.0
            .iter()
            .find(|connection| {
                connection.outputs.get(0) == Some(&Some(axis)) && connection.period == 1
            })
            .and_then(|connection| connection.inputs.get(0))
            .and_then(|d| *d)
    }

    pub fn unary_track_axis_down(&self, axis: usize) -> Option<usize> {
        self.0
            .iter()
            .find(|connection| {
                connection.inputs.get(0) == Some(&Some(axis)) && connection.period == 1
            })
            .and_then(|connection| connection.outputs.get(0))
            .and_then(|d| *d)
    }
}
