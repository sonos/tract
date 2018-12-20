use optim;

pub trait Context: std::fmt::Debug + Send + Sync {
    fn optimizer_passes(&self) -> Vec<Box<optim::OptimizerPass>>;
}

#[derive(Debug)]
pub struct DefaultContext;

impl Context for DefaultContext {
    fn optimizer_passes(&self) -> Vec<Box<optim::OptimizerPass>> {
        let mut passes = optim::normalization();
        passes.extend(optim::codegen().into_iter());
        passes
    }
}
