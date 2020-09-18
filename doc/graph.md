# Tract internals - Graph, Node, Fact and Op

These are a few notes about the way tract represents Neural Networks.

## Graph, Node, and OutletId

Neural Network natural structure is roughly a Directed Acyclic Graph.
In tract, the network structure is materialized mostly by Graph and BaseNode.

From core/src/model/{ graph, node }.rs, with some edits

```rust
pub struct Graph<F, O> {
    pub nodes: Vec<BaseNode<F, O>>,
    pub inputs: Vec<OutletId>,
    pub outputs: Vec<OutletId>,
    /* [...] */
}

pub struct BaseNode<F, O> {
    pub inputs: Vec<OutletId>,
    pub op: O,
    pub outputs: Vec<Outlet<F>>,
}

pub struct OutletId {
    pub node: usize,
    pub slot: usize,
}

pub struct Outlet<F: Fact + Hash> {
    pub fact: F,
    pub successors: Vec<InletId>,
}
```

`Graph` contains a list of `BaseNode` (the `nodes` field).

Each node contains an `Op`, which describes and implements the operator
the Node will apply to the data. Op can be a convolution, addition, etc. 
Each node has zero or more inputs referred (rarely) as *inlets*, and zero or
more outputs referred (often) as *outlets*. Most operators have one single
outlet.

When the network is ran, tract executes the nodes in the order induced by the
links: a node can only be ran all when tract knows the value of all its inputs.

Note that links between nodes (often called *wires*) are not symetrical: a
tensor produced by a node after its op is ran, can be fed to more than one
operator. An outlet can be connected to one or several inlets. On the other
hand one inlet can only have its value set by one outlet. A wire connects
*one single unique* outlet to several inlets.

An OutletId is a pair made of a node id and a *slot* which is the output number
of the Op. As most of the ops have a single output, the OutletId slot are
frequently zero.

As a consequence, OutletId is a the identifier for the wires in the graph,
connecting the designated outlet to successor inlets, and which tensor value
is set by the op from the node owning the outlet.

The `BaseNode` incoming wires are materialized by the `inputs` field, as a
simple vector.

For easier graph traversal, tract also maintains in `BaseNode` a redundant list
of outgoing wires: `outputs.successors` field. Additionaly, BaseNode stores a
`Fact` for each of its outgoing wires. Fact is the topic of the next section.

Finally, `Graph` owns two lists of OutletId designating the model inputs and
outputs: it is required that nodes pointed but the inputs' OutletId have the
specific Op `Source`. On the other hand, the output of a model can be any
node,

With this terminology, we can say that "running a network" involves finding
the value (a tensor) of each wire in the graph, 

    1. Source nodes have their value set by the network caller
    2. compute node outputs for unvisited nodes which have all their inputs
        known until all nodes are computed
    3. extract the model outputs from the wires pointed by `Graph.outputs`

## Facts

tract actually does much more than running the network as described in previous
section. It is capable of performing several optimisations, at the single node
or at the graph level. It can also perfom specific transformation, like
converting a streaming network to a pulsing network.

In order to perform these network rewrites, tract needs to be able to reason
about the *datum type* and the *shape* of each wire in the graph. Datum type,
despite the pedantic name, is just the type of each element of the tensor (like
f32, i8, or String).

Things get a bit more complicated here, because we often need to manipulate and
reason with graph with partial shape information. For instance, in TensorFlow
frozen network format, input shapes are usually not explicit, so we need to be
able to load graphs with partial facts before completing the shape and type
analysis.

`Fact` is a trait abstracting over the level of knowledge the `Graph` has about
the model types and shapes. The `F` type parameter in Graph is the kind of Fact
that *nodes* in the graph will contains as `outputs.fact`.

There are two major implementation of Fact in tract. The first one is
`TypedFact` in tract-core, the second one is `InferenceFact` in tract-hir.

From core/src/model/fact.rs with some edits.

```rust
pub struct TypedFact {
    pub datum_type: DatumType,
    pub shape: ShapeFact,
    pub konst: Option<Arc<Tensor>>,
}

pub struct ShapeFact(Vec<TDim>);
```

DatumType is a simple enumeration of the various element type a Tensor can hold
(like DatumType::F32, or DatumType::I8...). ShapeFact is a basically a vector
of TDim. Let's assume TDim is an integer for now.

TypedFact has also a optional constant value: if a tensor in the graph has a
constant value regardless of the network inputs, this is information the
optimiser may be able to use to simplify the network.

As we implied beforehand, this type is not suitable to reprensent network where
some wires have an unknown datum type, or partial shape information. In order
to reason about these networks, we need a more flexible Fact:

From hir/sr/infer/fact.rs

```rust
pub struct InferenceFact {
    pub datum_type: TypeFactoid,
    pub shape: ShapeFactoid,
    pub value: ValueFact,
}
```

*Inference* here takes the Computer Science meaning (as in "type inference"),
not the machine learning sense (synonym with *prediction*).

The InferenceFact main purpose is to support full shape and datum type
discovery of a graph. When this full *analysis* is done, the network facts
are all converted to TypedFact and the graph can undergo optimisation and
other transformations.

Without entering into too many details here, `TypeFactoid` is basically 
an `Option<DatumType>`, where the `None` value means "unknown". Similarly
`ShapeFact` can be seen as a `Vec<Option<TDim>>`, along with a boolean to
denote if the tensor rank (its number of dimension) is known or not.

## Fact inference, model pipeline and ops

The preferred way of running efficiently a network in tract is to make sure we
get it to a TypedModel (a model which Fact is a TypedFact), and let tract-core
optimise it.

For TensorFlow frozen networks, and to some extent for ONNX networks, the
analysis process will traverse the graph looking for incomplete shape or datum
type, and collaborate with the Op in the connecting nodes to "infer" more
information about the wires facts.

This collaboration is handled through the `InferenceOp` trait. All Ops used in
an InferenceModel must implement it. This is actually what the second type
parameter (`O`) of the Graph structure represent.

tract defines the `InferenceModel` type alias in hir/src/infer/mod.rs:

```rust
pub type InferenceModel = Graph<InferenceFact, Box<dyn InferenceOp>>;
```

InferenceOp interface is relatively complex to implement, as the code needs to
be able to operate on partial information. Once a model is "typed" we no longer
need this complex logic, but can work with a stricter, but easier to implement
variant of the InferenceOp contract, the `TypedOp`.

In core/src/model/types, we define:

```rust
pub type TypedModel = Graph<TypedFact, Box<dyn TypedOp>>;
```

The `TypedOp` contract is simple: an operation, given the TypedFacts (datum 
type, full shape, optional value) of all its input *must* be able to compute
the TypeFacts for its outputs.

in core/srs/ops/mod.rs:

```rust
trait TypedOp /* [...] */ {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TypedFact>>;
    /* [...] */
}
```

For most operations, implementing it is trivial. This simple contract is enough
to build a graph from the Source inputs (assuming we know their Fact) to its
outputs, discovering the full facts operation after operation.

The InferenceOp interface is a bit more involved, but allow to build the graph
in any aritrary order, leaving some facts partially determined, then running
the analyse to complete the missing bits.

```rust
trait InferenceOp /* [...] */ {
    fn infer(
        &mut self,
        inputs: TVec<&InferenceFact>,
        outputs: TVec<&InferenceFact>,
    ) -> TractResult<(Vec<InferenceFact>, Vec<InferenceFact>)> {
}
```

Here, the analyser will call the `infer` method providing all the known
information accumulated, and the op must do its best to return more determined
facts. Note that in the case of `TypedOp`, the typing information strictly goes
from inputs to outputs, whereas the `InferenceOp` allow operation to infer
inputs facts from output facts. This allows the framework to infer factoids
about model inputs, which can be useful when you're handed over a model without
explicit information about the input structure.

## EvalOp

Of course, the purpose of all of this is ultimately to *compute*. Once tract
has make sense of a model, it will just have to call the `eval` method on
each op in the graph in a determined order, keeping track of each tensor
already computed, handing them over to the successor ops.

```rust
pub trait EvalOp {
    fn eval(&self, inputs: Vec<Arc<Tensor>>) -> TractResult<Vec<Arc<Tensor>>>;
    /* .. */
}
```
