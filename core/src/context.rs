use optim;

pub trait Context: std::fmt::Debug {
    fn optimizer_passes(&self) -> Vec<Box<optim::OptimizerPass>>;
}

#[derive(Debug)]
pub struct DefaultContext;

impl Context for DefaultContext {
    fn optimizer_passes(&self) -> Vec<Box<optim::OptimizerPass>> {
        vec![Box::new(optim::PropConst), Box::new(optim::Reduce)]
    }
}
