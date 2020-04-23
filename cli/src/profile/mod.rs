use std::collections::HashMap;
use std::fmt::{Debug, Display};

use ansi_term::Color::*;

use tract_core::internal::*;
use tract_itertools::Itertools;

use crate::display_graph::DisplayOptions;
use crate::errors::*;
use crate::format::*;
use crate::rusage::Duration;
use crate::{Parameters, ProfilingMode};

mod regular;
//mod streaming;

#[derive(Debug, Default)]
pub struct ProfileData {
    pub nodes: HashMap<TVec<usize>, Duration>,
}

impl ProfileData {
    pub fn add(&mut self, node_id: &[usize], dur: Duration) -> ::tract_core::TractResult<()> {
        *self.nodes.entry(node_id.into()).or_insert(Duration::default()) += dur;
        Ok(())
    }

    pub fn sub(&mut self, node_id: &[usize], dur: Duration) -> ::tract_core::TractResult<()> {
        *self.nodes.entry(node_id.into()).or_insert(Duration::default()) -= dur;
        Ok(())
    }

    pub fn most_consuming_nodes(&self) -> CliResult<Vec<TVec<usize>>> {
        let top = self
            .nodes
            .iter()
            .sorted_by(|(_, a), (_, b)| {
                a.avg_real().partial_cmp(&b.avg_real()).unwrap_or(::std::cmp::Ordering::Greater)
            })
            .into_iter()
            .rev()
            .take(20)
            .map(|a| a.0.iter().cloned().collect())
            .collect();
        Ok(top)
    }

    fn op_name_for_id(model: &dyn Model, id: &[usize]) -> CliResult<String> {
        if id.len() == 1 {
            Ok(model.node_op(id[0]).name().into_owned())
        } else {
            let model = model.node_op(id[0]).as_typed().unwrap().nested_models()[0].1;
            Self::op_name_for_id(model, &id[1..])
        }
    }

    pub fn print_most_consuming_ops<F, O>(&self, model: &ModelImpl<F, O>) -> CliResult<()>
    where
        F: Fact + Clone + 'static,
        O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone + 'static + DynHash,
    {
        let sum = self.summed();
        println!("Most time consuming operations:");
        let mut operations = HashMap::new();
        let mut counters = HashMap::new();
        for (node, dur) in &self.nodes {
            let op_name = Self::op_name_for_id(model, node)?;
            *operations.entry(op_name.clone()).or_insert(Duration::default()) += *dur;
            *counters.entry(op_name).or_insert(0) += 1;
        }
        let mut operations: Vec<(&str, Duration)> =
            operations.iter().map(|(s, d)| (&**s, *d)).collect();
        operations.sort_by(|(_, a), (_, b)| {
            a.avg_real()
                .partial_cmp(&b.avg_real())
                .unwrap_or(::std::cmp::Ordering::Greater)
                .reverse()
        });
        for (operation, measure) in operations.iter().take(5) {
            println!(
                "{:20} {:3} nodes: {}",
                Blue.bold().paint(*operation),
                counters[&**operation],
                dur_avg_oneline_ratio(*measure, sum)
            );
        }
        Ok(())
    }

    pub fn summed(&self) -> Duration {
        let total_real = self.nodes.values().map(|n| n.avg_real()).sum();
        let total_sys = self.nodes.values().map(|n| n.avg_sys()).sum();
        let total_user = self.nodes.values().map(|n| n.avg_user()).sum();
        Duration { total_real, total_sys, total_user }
    }
}

/// Handles the `profile` subcommand.
pub fn handle(
    params: &Parameters,
    profiling: ProfilingMode,
    display_options: DisplayOptions,
    monitor: Option<&readings_probe::Probe>,
) -> CliResult<()> {
    match &profiling {
        ProfilingMode::Regular { .. } => regular::handle(params, profiling, display_options),
        ProfilingMode::RegularBenching { .. } => regular::handle_benching(params, profiling, monitor),
    }
}
