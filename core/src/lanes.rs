use crate::internal::*;

/// The lanes of one laned state: which are taken, and which of them a turn
/// seats.
///
/// Plain data. Taking a lane does not touch the state's buffers, and clearing
/// what a stream left in a lane it gave up is the table's caller's, since it
/// writes the state -- device memory for a state on a GPU -- and must run where
/// the state lives. So a lane handed to a new stream carries the previous one's
/// history until that caller resets it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneTable {
    taken: Vec<bool>,
}

impl LaneTable {
    pub fn new(max_lanes: usize) -> TractResult<LaneTable> {
        ensure!(max_lanes > 0, "A laned state needs at least one lane");
        Ok(LaneTable { taken: vec![false; max_lanes] })
    }

    /// The extent of the lane axis of the state's per-lane buffers, fixed for
    /// the life of the state.
    pub fn max_lanes(&self) -> usize {
        self.taken.len()
    }

    pub fn taken(&self) -> usize {
        self.taken.iter().filter(|t| **t).count()
    }

    /// The lowest free lane, `None` when every lane is taken -- whether that
    /// blocks the new stream or fails it is the caller's policy. Lowest first,
    /// so that a turn seating every lane seats a run of consecutive lanes.
    pub fn take(&mut self) -> Option<LaneId> {
        let lane = self.taken.iter().position(|t| !t)?;
        self.taken[lane] = true;
        Some(LaneId(lane))
    }

    /// Hand `lane` back, for [`LaneTable::take`] to give to another stream.
    pub fn give_back(&mut self, lane: LaneId) -> TractResult<()> {
        ensure!(self.is_taken(lane), "Lane {} is not taken, so it can not be given back", lane.0);
        self.taken[lane.0] = false;
        Ok(())
    }

    pub fn is_taken(&self, lane: LaneId) -> bool {
        self.taken.get(lane.0).copied().unwrap_or(false)
    }

    /// Seat `lanes`, in that order: seat `ix` of the coming turn carries the
    /// `ix`th of them. Every one must be taken, so that a stream which ended
    /// can not be seated by a stale handle of it.
    pub fn seat(&self, lanes: impl IntoIterator<Item = LaneId>) -> TractResult<Seating> {
        let lanes: Vec<LaneId> = lanes.into_iter().collect();
        for lane in &lanes {
            ensure!(self.is_taken(*lane), "Seating lane {}, which no stream took", lane.0);
        }
        Seating::new(self.max_lanes(), lanes)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn takes_the_lowest_free_lane() -> TractResult<()> {
        let mut table = LaneTable::new(3)?;
        assert_eq!(table.take(), Some(LaneId(0)));
        assert_eq!(table.take(), Some(LaneId(1)));
        table.give_back(LaneId(0))?;
        assert_eq!(table.take(), Some(LaneId(0)));
        assert_eq!(table.taken(), 2);
        Ok(())
    }

    #[test]
    fn runs_out_of_lanes() -> TractResult<()> {
        let mut table = LaneTable::new(1)?;
        assert_eq!(table.take(), Some(LaneId(0)));
        assert_eq!(table.take(), None);
        Ok(())
    }

    #[test]
    fn gives_back_a_taken_lane_only() -> TractResult<()> {
        let mut table = LaneTable::new(2)?;
        assert!(table.give_back(LaneId(0)).is_err());
        table.take();
        table.give_back(LaneId(0))?;
        assert!(table.give_back(LaneId(0)).is_err());
        assert!(table.give_back(LaneId(7)).is_err());
        Ok(())
    }

    #[test]
    fn seats_taken_lanes_in_order() -> TractResult<()> {
        let mut table = LaneTable::new(4)?;
        table.take();
        table.take();
        table.take();
        table.give_back(LaneId(1))?;
        let seating = table.seat([LaneId(2), LaneId(0)])?;
        assert_eq!(seating.max_lanes(), 4);
        assert_eq!(seating.occupancy(), 2);
        assert_eq!(seating.address(0), (Some(0), Some(2)));
        assert_eq!(seating.address(1), (Some(1), Some(0)));
        assert!(table.seat([LaneId(0), LaneId(1)]).is_err());
        assert!(table.seat([LaneId(0), LaneId(0)]).is_err());
        Ok(())
    }
}
