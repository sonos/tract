use std::collections::HashMap;
use rusage::Duration;
use tfdeploy::tfpb::graph::GraphDef;
use colored::Colorize;

use tfdeploy::{Model, ModelState, Node};
use format::*;
use errors::*;

use { Parameters, ProfilingMode };

mod regular;
mod streaming;

#[derive(Debug)]
pub struct ProfileData {
    pub nodes: HashMap<usize, Duration>,
}

impl ProfileData {
    pub fn new(model: &Model) -> ProfileData {
        ProfileData {
            nodes: HashMap::with_capacity(model.nodes.len()),
        }
    }

    pub fn add(&mut self, node:&Node, dur: Duration) -> ::tfdeploy::Result<()> {
        *self.nodes.entry(node.id).or_insert(Duration::default()) += dur;
        Ok(())
    }

    pub fn print_most_consuming_nodes(&mut self, model: &Model, graph: &GraphDef, state: Option<&ModelState>) -> Result<()> {
        let sum = self.summed();
        println!("Most time consuming nodes:");
        let mut nodes:Vec<(usize, Duration)> = self.nodes.iter().map(|(&a, &b)| (a,b)).collect();
        nodes.sort_by(|(_, a), (_, b)| a.avg_real().partial_cmp(&b.avg_real()).unwrap_or(::std::cmp::Ordering::Greater).reverse());
        for (node, measure) in nodes.iter().take(5) {
            let node = model.get_node_by_id(*node)?;
            print_node(
                node,
                graph,
                state,
                vec![dur_avg_oneline_ratio(*measure, sum)],
                vec![]
            );
        }
        Ok(())
    }

    pub fn print_most_consuming_ops(&self, model: &Model) -> Result<()> {
        let sum = self.summed();
        println!("Most time consuming operations:");
        let mut operations = HashMap::new();
        let mut counters = HashMap::new();
        for (node, dur) in &self.nodes {
            let node = model.get_node_by_id(*node)?;
            let mut cell = operations.entry(node.op_name.to_string()).or_insert(Duration::default());
            // do not use duration addition here, as we are summing for real
            // instead of averaging
            cell.total_real += dur.avg_real();
            cell.total_sys += dur.avg_sys();
            cell.total_user += dur.avg_user();
            cell.counter = 1;
            *counters.entry(node.op_name.to_string()).or_insert(0) += 1;
        }
        let mut operations:Vec<(&str, Duration)> = operations.iter().map(|(s,d)| (&**s, *d)).collect();
        operations.sort_by(|(_, a), (_, b)|
            a.avg_real().partial_cmp(&b.avg_real()).unwrap_or(::std::cmp::Ordering::Greater).reverse()
        );
        for (operation, measure) in operations.iter().take(5) {
            println!(
                "{:20} {:3} calls: {}",
                operation.blue().bold(), counters[&**operation], dur_avg_oneline_ratio(*measure, sum)
                );
        }
        Ok(())
    }

    pub fn summed(&self) -> Duration {
        let total_real = self.nodes.values().map(|n| n.avg_real()).sum();
        let total_sys = self.nodes.values().map(|n| n.avg_sys()).sum();
        let total_user = self.nodes.values().map(|n| n.avg_user()).sum();
        Duration {
            total_real, total_sys, total_user, counter: 1
        }
    }
}

/// Handles the `profile` subcommand.
pub fn handle(params: Parameters, profiling:ProfilingMode) -> Result<()> {
    match &profiling {
        ProfilingMode::Regular{..} => regular::handle(params, profiling),
        ProfilingMode::RegularBenching{..} => regular::handle_benching(params, profiling),
        ProfilingMode::StreamCruising => streaming::handle_cruise(params),
        ProfilingMode::StreamBuffering => streaming::handle_buffering(params),
        ProfilingMode::StreamBenching{..} => streaming::handle_bench(params, profiling),
    }
}

