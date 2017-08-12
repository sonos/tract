use std::path;

use tfdeploy::{Matrix, GraphAnalyser};
use errors::*;

pub struct TfDeploy {
    graph: GraphAnalyser,
}

pub fn build<P: AsRef<path::Path>>(p: P) -> Result<Box<::TfExecutor>> {
    let graph = GraphAnalyser::from_file(p)?;
    Ok(Box::new(TfDeploy { graph }))
}

impl ::TfExecutor for TfDeploy {
    fn run(&mut self, inputs: Vec<(&str, Matrix)>, output_name: &str) -> Result<Vec<Matrix>> {
        for input in inputs {
            self.graph.set_value(input.0, input.1)?;
        }
        Ok(self.graph.eval(output_name).map(|v| v.clone())?)
    }
}
