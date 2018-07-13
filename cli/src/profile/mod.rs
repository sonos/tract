use std::collections::HashMap;
use rusage::Duration;
use simplelog::Level::Info;
use tfdeploy::tfpb::graph::GraphDef;

use tfdeploy::{Model, ModelState, Node};
use format::*;
use errors::*;

use { Parameters, ProfilingParameters };

mod regular;
mod streaming;

pub struct ProfileData {
    pub global: Duration,
    pub nodes: HashMap<usize, Duration>,
    pub operations: HashMap<String, (Duration, usize)>,
}

impl ProfileData {
    pub fn new(model: &Model) -> ProfileData {
        let mut operations = HashMap::new();
        for node in &model.nodes {
            operations.insert(node.op_name.to_string(), (Duration::default(), 0));
        }
        ProfileData {
            global:Duration::new(),
            nodes: HashMap::with_capacity(model.nodes.len()),
            operations
        }
    }

    pub fn add(&mut self, node:&Node, dur: Duration) -> ::tfdeploy::Result<()> {
        self.global += dur;
        *self.nodes.entry(node.id).or_insert(Duration::default()) += dur;
        let ref mut pair = self.operations.get_mut(&node.op_name).unwrap(); // pre-filled in new
        pair.0 += dur;
        pair.1 += 1;
        Ok(())
    }

    pub fn print_most_consuming_nodes(&mut self, model: &Model, graph: &GraphDef, state: Option<&ModelState>) -> Result<()> {
        println!("Most time consuming nodes:");
        let mut nodes:Vec<(usize, Duration)> = self.nodes.iter().map(|(&a, &b)| (a,b)).collect();
        nodes.sort_by(|(_, a), (_, b)| a.avg_real.partial_cmp(&b.avg_real).unwrap().reverse());
        for (node, measure) in nodes.iter().take(5) {
            let node = model.get_node_by_id(*node)?;
            print_node(
                node,
                graph,
                state,
                vec![dur_avg_oneline_ratio(*measure, self.global)],
                vec![]
            );
        }

        println!();
        println!("Total execution time (for {} nodes): {}", self.nodes.len(), dur_avg_oneline(self.global));
        Ok(())
    }

    pub fn print_most_consuming_ops(&self) {
        use colored::Colorize;

        println!("Most time consuming operations:");
        let mut operations = self.operations.iter()
            .map(|(o, (measure, c))| (o, measure, c))
            .collect::<Vec<_>>();
        operations.sort_by(|(_, a, _), (_, b, _)| a.avg_real.partial_cmp(&b.avg_real).unwrap().reverse());
        for (operation, measure, count) in operations.iter().take(5) {
            println!(
                "{:20} {:3} calls: {}",
                operation.blue().bold(), count, dur_avg_oneline_ratio(**measure, self.global)
            );
        }
    }
}

/// Handles the `profile` subcommand.
pub fn handle(params: Parameters, profiling:ProfilingParameters) -> Result<()> {
    if params.input.as_ref().unwrap().shape.iter().all(|dim| dim.is_some()) {
        regular::handle(params, profiling)
    } else {
        if profiling.buffering {
            streaming::handle_buffering(params)
        } else {
            streaming::handle_cruise(params, profiling)
        }
    }
}

