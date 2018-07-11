use std::collections::HashMap;
use rusage::Duration;
use simplelog::Level::Info;
use tfdeploy::tfpb::graph::GraphDef;

use tfdeploy::*;
use format::*;

pub struct ProfileData<'a> {
    graph: &'a GraphDef,
    model: &'a Model,
    pub global: Duration,
    pub nodes: HashMap<usize, Duration>,
    pub operations: HashMap<&'a str, (Duration, usize)>,
}

impl<'a> ProfileData<'a> {
    pub fn new(graph: &'a GraphDef, model: &'a Model) -> ProfileData<'a> {
        let capacity = model.nodes().len();
        ProfileData {
            graph,
            model,
            global:Duration::new(),
            nodes: HashMap::with_capacity(capacity),
            operations:HashMap::with_capacity(capacity),
        }
    }

    pub fn add(&mut self, node_id:usize, dur: Duration) -> Result<()> {
        self.global += dur;
        *self.nodes.entry(node_id).or_insert(Duration::default()) += dur;
        let node = self.model.get_node_by_id(node_id)?;
        let ref mut pair = self.operations.entry(&node.op_name).or_insert((Duration::default(),0));
        pair.0 += dur;
        pair.1 += 1;
        Ok(())
    }

    pub fn print_most_consuming_nodes(&mut self, state: Option<&ModelState>) -> Result<()> {
        use colored::Colorize;

        println!("Most time consuming nodes:");
        let mut nodes:Vec<(usize, Duration)> = self.nodes.iter().map(|(&a, &b)| (a,b)).collect();
        nodes.sort_by(|(_, a), (_, b)| a.avg_real.partial_cmp(&b.avg_real).unwrap().reverse());
        for (node, measure) in nodes.iter().take(5) {
            let node = self.model.get_node_by_id(*node)?;
            print_node(
                node,
                &self.graph,
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
                "- {:20} {:3} nodes: {}",
                operation.blue().bold(), count, dur_avg_oneline_ratio(**measure, self.global)
            );

            if log_enabled!(Info) {
                println!("    - {:.3} ms in total.", measure.total_real * 1e3);
            }
        }
    }
}

