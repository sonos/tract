use tensorflow::Graph;
use tensorflow::Session;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;

pub struct Tensorflow {
    session: Session,
    graph: Graph,
}
