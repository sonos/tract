use crate::rusage::Duration;
use ansi_term::Color::*;
use std::collections::HashMap;

use crate::errors::*;
use crate::format::*;
use itertools::Itertools;
use tract_core::internal::*;

use crate::display_graph::DisplayOptions;
use crate::{Parameters, ProfilingMode, SomeGraphDef};

mod regular;
//mod streaming;

#[derive(Debug, Default)]
pub struct ProfileData {
    pub nodes: HashMap<usize, Duration>,
}

impl ProfileData {

    pub fn add<TI: TensorInfo>(
        &mut self,
        node: &Node<TI>,
        dur: Duration,
    ) -> ::tract_core::TractResult<()> {
        *self.nodes.entry(node.id).or_insert(Duration::default()) += dur;
        Ok(())
    }

    pub fn most_consuming_nodes(&self) -> CliResult<Vec<usize>> {
        let top = self
            .nodes
            .iter()
            .sorted_by(|(_, a), (_, b)| {
                a.avg_real().partial_cmp(&b.avg_real()).unwrap_or(::std::cmp::Ordering::Greater)
            })
            .into_iter()
            .rev()
            .take(5)
            .map(|a| *a.0)
            .collect();
        Ok(top)
    }

    pub fn print_most_consuming_ops<TI: TensorInfo>(&self, model: &Model<TI>) -> CliResult<()> {
        let sum = self.summed();
        println!("Most time consuming operations:");
        let mut operations = HashMap::new();
        let mut counters = HashMap::new();
        for (node, dur) in &self.nodes {
            let node = &model.nodes()[*node];
            let mut cell =
                operations.entry(node.op.name().to_string()).or_insert(Duration::default());
            // do not use duration addition here, as we are summing for real
            // instead of averaging
            cell.total_real += dur.avg_real();
            cell.total_sys += dur.avg_sys();
            cell.total_user += dur.avg_user();
            cell.counter = 1;
            *counters.entry(node.op.name().to_string()).or_insert(0) += 1;
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
                "{:20} {:3} calls: {}",
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
        Duration { total_real, total_sys, total_user, counter: 1 }
    }
}

/// Handles the `profile` subcommand.
pub fn handle(
    params: Parameters,
    profiling: ProfilingMode,
    display_options: DisplayOptions,
) -> CliResult<()> {
    match &profiling {
        ProfilingMode::Regular { .. } => regular::handle(params, profiling, display_options),
        ProfilingMode::RegularBenching { .. } => regular::handle_benching(params, profiling),
    }
}
