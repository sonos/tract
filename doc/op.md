# Anatomy of an Op

Operators, Op for short are central players in tract. They are of course
responsible for doing the actual computations and transformations on the
tensor, but also must collaborate with the loading and optimisation
framework, and sometimes also with their peers to analyse, validate or reduce
the model to a simple form.

These tasks are varying widely in complexity depending on the actual op, and
how much effort have been put into optimising them.

tract defines, in `core` and `hir` a few traits that an operation can or must
implement.

## tract-core Op trait

This trait contains the minimal metadata that all ops will share.

```rust
pub trait Op:
    fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + EvalOp + DynHash
{
    fn name(&self) -> Cow<str>;

    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    fn same_as(&self, _other: &dyn Op) -> bool {
        false
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![])
    }

    fn as_typed(&self) -> Option<&dyn TypedOp>;
}
```

`name()` and `info()` are mostly useful for debugging and
auditing (the command line interface and error messages will use these entry
points).

`validation` is a hint, for debugging and auditing purposed that this
operator is likely to generates rounding errors. If we run network side to side
with a reference implementation (TensorFlow for example) do we expect the
results to be exactly the same, to the last bit, or should we use a more
lenient way or comparing the results.

We rely a lot on Rust's *trait object* mecanism around Ops. It can be
surprising to the new comer that things such as Clone become complicated in the
context of trait objects, but their are a few crates that offer workaround:
tract Op leverage `dyn_clone` crate functionality, mimicks it with the DynHash
trait and macros to be able to get a hash of an `&dyn Op`. `same_as` is another
workaround (maybe a missed opportunity for a DynHash-like mecanism actualy)
as PartialEq is incompatible with trait objects.

`as_typed()` allows the framework to dynamicaly ask the Op to cast itself to
&TypedOp. rust offers no "QueryInterface"-like mecanism to switch from trait to
trait in a family of trait, so there are a few casting methods here and there
in tract op traits. Most of them are implemented using trivial functions
(`return Some(&self);`) that we generate with macros.

We also need `Op` to implement Downcast. There are actually quite a few
situations in tract where we need to Downcast, like implementing `same_as()`.
As we said, Op pushes Rust dynamic typing where it does not like to go :)

## tract-core EvalOp trait

While Op is mostly metadata, EvalOp is at the other end of spectrum with the
business side of things.

```rust
pub trait EvalOp {
     fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
         bail!("stateless evaluation not implemented")
     }

     fn state(
         &self,
         session: &mut SessionState,
         node_id: usize,
     ) -> TractResult<Option<Box<dyn OpState>>> {
         Ok(None)
     }

     fn is_stateless(&self) -> bool;
}
```

The EvalOp realize the actual computation the Operator is supposed to perform. It
supports both *stateful* and *stateless* operators. Most of them are stateless:
they should just implement `eval` method and say so in `is_stateless()`. The
handful of stateful operators will implements `state()` instead and return
`false` is is_stateless: the framework will call `state()` during the network
initialization, then will call `eval()` on the obtained `OpState` instead:

```rust
pub trait OpState: fmt::Debug + Send + dyn_clone::DynClone {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>>;
}
```

Here the eval implementation is free to mute some operation internal state if
required, or access the `SessionState`.

But most operators are stateless anyway.

## tract-core TypedOp trait

`Op` is metadata, `EvalOp` is runtime, `TypedOp` is about reasoning on the
model. Most optimisations and transformations will operate on TypedOp
implementors. TypedOp has a minimal handful of methods that are required to be
implemented and many optional. While they are not strictly required, a missing,
or partial implementation may prevent the optimiser to perform optimisations
that require an Op to "collaborate" with its peers.

```rust
pub trait TypedOp:
    Op + fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + EvalOp + DynHash
{
    fn as_op(&self) -> &dyn Op;
    fn as_op_mut(&mut self) -> &mut dyn Op;

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>>;

    /*[...]*/
}
```

First we have two more cross-trait casting methods. Once again, we have macros
that do a trivial return self.

Then comes `output_facts()`. This is a pivotal method in TypedOp. As discussed
in the [graph write-up](graph.md) it is the lighter version of the type
inference, just enough to build networks from source to outputs while
maintaining known shapes and data types.

When building a `TypedModel`, we repeatedly create new `TypedOp`s, retrieving
from the network the `TypedFact` for their inputs.  Then output_facts() is
called, and the TypedOp must return the output facts.  The framework can then
create the `TypedNode` and append it to the partial TypedModel before repeating
the process. This also makes output_fact() a great place to run as many
validation checks and detect inconsistency soon instead of discovering them at
runtime (in eval).

## trait-hir InferenceOp trait

Sitting between the tract-core and the training frameworks loaders, tract-hir
contains everything needed to translate networks from ONNX or TensorFlow to
tract-core. Tract-hir role is to map the training framworks' very expressive
operators set to the more constrained tract-core. In order to perform this
translation, the partial type information contained in ONNX or TensorFlow
protobuf files must be used to infer the full types and shapes of the network.

The contract here is significantly more complex than the one required by
TypedOp.

```rust
pub trait InferenceOp:
    Op + fmt::Debug + tract_core::dyn_clone::DynClone + Send + Sync + 'static + Downcast + EvalOp + DynHash
{
    fn infer(
        &mut self,
        inputs: TVec<&InferenceFact>,
        outputs: TVec<&InferenceFact>,
        observed: TVec<&InferenceFact>,
    ) -> TractResult<(TVec<InferenceFact>, TVec<InferenceFact>, TVec<InferenceFact>)>;
    /*[...]*/

    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
  ) -> TractResult<TVec<OutletId>>;
}
```

The InferenceFact is a partially determined version of TypedFact: typically,
TypedFact has a DatumType field to represent the type of the tensor element,
while InferenceFact datum_type is akin to an `Option<DatumType>` where `None`
means unknown. In a similar fashion, the shape can be partially known, with
rank determined or not and individual dimensions known or not.

The framework will try to propagate type information accross the graph,
refining incrementally its knowledge of all the inference facts. It will do
so by calling the `infer()` method on operators which interfaces are not fully
determined. The operator receives as paramaters the current information on its
inputs and outputs, try to improve them and returns the refined versions. The
third paramaters and result (`observed`) is out of scope here.

Once a network has been entirely typed, it can be translated to a TypedOp. The
framework will visit the entire network and call the `to_typed()` method on
each operator. The operator is responsible to "wire" itself into the target
TypedModel (creating a node) and return its output(s) wires.

## trait-hir InferenceRulesOp trait

Implementing the `infer()` method by hand is tedious, mostly because
type information constraint can flow in both directions, from an input to
output, but also the other way around, or even between two inputs. We
developped an alternative, which makes the process more declarative. It is a
syntax heavy, but less error-prone once rustc is happy.

```rust
trait InferenceOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()>;
    /*[...]*/
}
```

Here, we can use a more declarative approach and load the "solver" with rules
specifying relations between inputs and outputs. This make it possible to
propagate information in all directions with one single rule.

The rules solver syntax is a bit old and arcane, and could certainly be
improved or simplified, but it is still much easier to use than writing rules
by hand.

## Loading from the frameworks

Training frameworks (TensorFlow and Onnx) use protobuf as a serialization
format. tract-tensorflow (and tract-onnx) can read these and build the neural 
network as an InferenceModel in memory. When the framework parses a node, the
operation type is manifested by its name. Then the way to interpret and plug-in
the various attribute depends on the Operator itself.

When loading the framework object, tract builds a mapping of
operator names to operator constructor functions that is responsible for
extracting the attributes from the parsed protobuf.

Modules containing operators typically expose a register_all_ops function that
feeds this map. Here is an example from `onnx/src/ops/nn/mod.rs`.

```rust
pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("ArgMax", arg_max_min);
    reg.insert("ArgMin", arg_max_min);
    reg.insert("AveragePool", average_pool);
    reg.insert("BatchNormalization", batch_normalization);
    /* [...] */
}
```

The `batch_normalization` function looks like this:

```rust
pub fn batch_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5);
    let spatial = node.get_attr_opt("spatial")?.unwrap_or(1);
    if spatial != 1 {
        bail!("BatchNormalization: attribute 'spatial' is not supported (deprecated by ONNX operator set 9)")
    }
    Ok((expand(batch_norm::BatchNorm::new(nn::DataFormat::NCHW, epsilon, spatial != 0)), vec![]))
}
```

It is given the protobuf parsed Node, and extracts two attributes from it.
Then it buidls an actual operation (as an Expansion, more on this later).

## Dumping as OPL, loading from OPL

"Good citizen" operators know how to dump themselves in OPL, and load from
them.

OPL is tract NNEF-based format. Some operators are compatible with NNEF, in
which case they can use NNEF standard form, but many operators from ONNX and
TensorFlow can only be handled with extensions.

NNEF compatible operators are dumped and loaded by code in 
`nnef/src/ops/nnef/mod.rs`.

Each OPL module (`nnef`, `pulse-opl`, `onnx-opl`) defines a registry of
operators, containing both OPL loaders and OPL dumpers.

```rust
pub struct Registry {
    pub id: String,
    pub fragments: HashMap<String, FragmentDef>,
    pub primitives: HashMap<String, (Vec<ast::Parameter>, ToTract)>,
    pub unit_element_wise_ops: Vec<(String, Box<dyn ElementWiseMiniOp>)>,
    pub element_wise_ops: Vec<(String, TypeId, FromTract, Vec<ast::Parameter>, ToTract)>,
    pub binary_ops: Vec<(String, Box<dyn BinMiniOp>)>,
    pub from_tract: HashMap<TypeId, FromTract>,
}
pub type ToTract = fn(&mut ModelBuilder, &ResolvedInvocation) -> TractResult<TVec<OutletId>>;
pub type FromTract = fn(&mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>>;
```

The generic dumping mecanism relies on `from_tract` HashMap: it maps a rust
TypeId (the one for the TypedOp we need to dump) to a FromTract dumping
function. The mutable IntoAst is modified by the callback to store the
representation of the Op. The callback can add NNEF fragments (NNEF lingo for
functions) to the NNEF document but its main responsibility is to translate 
the node and its op to some NNEF ast nodes.

## Expansions, and rules wrapper
