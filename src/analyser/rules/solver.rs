use std::ops::{Add, Neg};
use std::fmt;

use num::Zero;

use analyser::prelude::*;
use analyser::rules::prelude::*;
use model::TVec;
use Result;



/// A structure that holds the current sets of TensorFacts.
///
/// This is used during inference (see `Solver::infer`) to let rules compute
/// the value of expressions which involve tensor properties.
#[derive(Debug, new)]
pub struct Context {
    pub inputs: TVec<TensorFact>,
    pub outputs: TVec<TensorFact>,
}

impl Context {
    /// Returns the current value of the variable at the given path.
    pub fn get<T: Output>(&self, path: &Path) -> Result<T> {
        let value = get_path(self, &path[..])?;

        Ok(T::from_wrapped(value)?)
    }

    /// Tries to set the value of the variable at the given path.
    pub fn set<T: Output>(&mut self, path: &Path, value: T) -> Result<()> {
        set_path(self, &path[..], T::into_wrapped(value))?;

        Ok(())
    }
}

/// A rule that can be applied by the solver.
pub trait Rule<'rules>: fmt::Debug {
    /// Tries to apply the rule to a given context.
    ///""
    /// The method must return Ok(true) if the rule was applied successfully
    /// (meaning that the Context was mutated), or Ok(false) if the rule was
    /// not applied but didn't generate any errors.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule<'rules> + 'rules>>)>;

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path>;
}

/// The `equals` rule.
/// It states that the given expressions must all be equal.
///
/// It can be added to the solver via the following two methods:
/// ```text
/// solver.equals(a, b);
/// solver.equals_all(vec![a, b, ...]);
/// ```
struct EqualsRule<T: Output + Fact> {
    items: Vec<Exp<T>>,
}

impl<T: Output + Fact> EqualsRule<T> {
    /// Creates a new EqualsRule instance.
    pub fn new(items: Vec<Exp<T>>) -> EqualsRule<T> {
        EqualsRule { items }
    }
}

impl<'rules, T: Output + Fact> Rule<'rules> for EqualsRule<T> {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule<'rules> + 'rules>>)> {
        let mut value = None;
        for item in &self.items {
            let v = item.get(context)?;
            if v.is_concrete() {
                value = Some(v);
                break;
            }
        }
        if let Some(value) = value {
            let mut changed = false;
            for item in &self.items {
                changed |= item.set(context, value.clone())?;
            }
            return Ok((changed, vec![]));
        }
        Ok((false, vec![]))
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.items.iter().flat_map(|e| e.get_paths()).collect()
    }
}

impl<'rules, T: Output + Fact> fmt::Debug for EqualsRule<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.items[0])?;
        for item in &self.items[1..] {
            write!(formatter, " == {:?}", item)?;
        }
        Ok(())
    }
}

/// The `equals_zero` rule.
/// It states that the given expression must equal zero.
///
/// It can be added to the solver via the following method:
/// ```text
/// solver.equals_zero(vec![a, b, ...]);
/// ```
struct EqualsZeroRule<F>(Exp<F>)
where
    F: Fact + Zero + Add<F, Output = F> + Neg<Output = F> + Clone + ::std::fmt::Debug + Output;

impl<'rules, F> Rule<'rules> for EqualsZeroRule<F>
where
    F: Fact + Zero + Add<F, Output = F> + Neg<Output = F> + Clone + ::std::fmt::Debug + Output,
{
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule<'rules> + 'rules>>)> {
        Ok((self.0.set(context, F::zero())?, vec![]))
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.0.get_paths()
    }
}

impl<F> fmt::Debug for EqualsZeroRule<F>
where
    F: Fact + Zero + Add<F, Output = F> + Neg<Output = F> + Clone + ::std::fmt::Debug + Output,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(formatter)?;
        write!(formatter, " == 0")
    }
}

/// The `with` rule.
/// It allows you to add more rules to the solver using what is known about an
/// expression.using a closure that takes the value as parameter.
///
/// It can be added to the solver via the following method:
/// ```text
/// solver.with(input.rank, |solver, ir|
///     // Add more rules to `solver` here.
/// );
/// ```
pub struct WithRule<'rules, T: Fact> {
    pub item: Exp<T>,
    pub closure: Box<Fn(&mut Solver<'rules>, T) + 'rules>,
}

impl<'rules, T: Output + Fact> WithRule<'rules, T> {
    /// Creates a new GivenRule instance.
    pub fn new<F>(item: Exp<T>, closure: F) -> WithRule<'rules, T>
    where
        F: Fn(&mut Solver<'rules>, T) + 'rules,
    {
        let closure = Box::new(closure);
        WithRule { item, closure }
    }
}

impl<'rules, T: Output + Fact> Rule<'rules> for WithRule<'rules, T> {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule<'rules> + 'rules>>)> {
        let value = self.item.get(context)?;
        trace!("    With rule: {:?} is {:?}", self.item, value);
        let mut solver = Solver::default();
        (self.closure)(&mut solver, value);
        Ok((true, solver.take_rules()))
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.item.get_paths()
    }
}

impl<'s, T: Output + Fact> fmt::Debug for WithRule<'s, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "WithRule {{ {:?} }}", self.item)
    }
}

/// The `given` rule.
/// It allows you to add more rules to the solver once the value of a given
/// expression is known, using a closure that takes the value as parameter.
///
/// It can be added to the solver via the following method:
/// ```text
/// solver.given(input.rank, |solver, ir|
///     // Add more rules to `solver` here.
/// );
/// ```
pub struct GivenRule<'rules, T: Fact> {
    pub item: Exp<T>,
    pub closure: Box<Fn(&mut Solver<'rules>, T::Concrete) + 'rules>,
}

impl<'rules, T: Output + Fact> GivenRule<'rules, T> {
    /// Creates a new GivenRule instance.
    pub fn new<F>(item: Exp<T>, closure: F) -> GivenRule<'rules, T>
    where
        F: Fn(&mut Solver<'rules>, T::Concrete) + 'rules,
    {
        let closure = Box::new(closure);

        GivenRule { item, closure }
    }
}

impl<'rules, T: Output + Fact> Rule<'rules> for GivenRule<'rules, T> {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule<'rules> + 'rules>>)> {
        let value = self.item.get(context)?;

        if let Some(value) = value.concretize() {
            trace!("    Given rule: {:?} is {:?}", self.item, value);
            // We create a new solver instance, which will be populated with
            // new rules by the code inside the closure.
            let mut solver = Solver::default();

            (self.closure)(&mut solver, value);

            Ok((true, solver.take_rules()))
        } else {
            trace!(
                "In {:?}, failed to convert {:?} to expected type",
                self,
                self.item.get(context)?.wrap()
            );
            Ok((false, vec![]))
        }
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.item.get_paths()
    }
}

impl<'s, T: Output + Fact> fmt::Debug for GivenRule<'s, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GivenRule {{ {:?} }}", self.item)
    }
}

/// The `given` rule.
/// It allows you to add more rules to the solver once the value of a given
/// expression is known, using a closure that takes the value as parameter.
///
/// It can be added to the solver via the following method:
/// ```text
/// solver.given(input.rank, |solver, ir|
///     // Add more rules to `solver` here.
/// );
/// ```
pub struct GivenAllRule<'rules, T: Fact> {
    pub items: Vec<Exp<T>>,
    pub closure: Box<Fn(&mut Solver<'rules>, Vec<T::Concrete>) + 'rules>,
}

impl<'rules, T: Output + Fact> GivenAllRule<'rules, T> {
    /// Creates a new GivenRule instance.
    pub fn new<F>(items: Vec<Exp<T>>, closure: F) -> GivenAllRule<'rules, T>
    where
        F: Fn(&mut Solver<'rules>, Vec<T::Concrete>) + 'rules,
    {
        let closure = Box::new(closure);

        GivenAllRule { items, closure }
    }
}

impl<'rules, T: Output + Fact> Rule<'rules> for GivenAllRule<'rules, T> {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule<'rules> + 'rules>>)> {
        let values: Vec<T> = self.items
            .iter()
            .map(|it| it.get(context))
            .collect::<Result<Vec<T>>>()?;
        let concrete: Vec<_> = values.iter().filter_map(|it| it.concretize()).collect();

        if concrete.len() == self.items.len() {
            trace!("    Given all rule: {:?} is {:?}", self.items, values);
            // We create a new solver instance, which will be populated with
            // new rules by the code inside the closure.
            let mut solver = Solver::default();
            (self.closure)(&mut solver, concrete);
            Ok((true, solver.take_rules()))
        } else {
            Ok((false, vec![]))
        }
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.items.iter().flat_map(|it| it.get_paths()).collect()
    }
}

impl<'s, T: Output + Fact> fmt::Debug for GivenAllRule<'s, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GivenAllRule {:?}", self.items)
    }
}

/// A declarative constraint solver for tensors.
#[derive(Default)]
pub struct Solver<'rules> {
    // The rules used by the solver.
    pub rules: Vec<Box<Rule<'rules> + 'rules>>,
}

impl<'rules> Solver<'rules> {
    /// Consumes the solver and returns the rules that it uses.
    pub fn take_rules(self) -> Vec<Box<Rule<'rules> + 'rules>> {
        self.rules
    }

    /// Runs the solver on a set of TensorFacts.
    ///
    /// This method returns:
    /// - Err(_) if a constraint couldn't be satisfied.
    /// - Ok(None) if no more information about tensors could be deduced.
    /// - Ok(Some(facts)) otherwise, with `facts` the new TensorFacts.
    pub fn infer(
        self,
        mut facts: (TVec<TensorFact>, TVec<TensorFact>),
    ) -> Result<(TVec<TensorFact>, TVec<TensorFact>)> {
        for f in &mut facts.0 {
            f.reduce();
        }
        for f in &mut facts.1 {
            f.reduce();
        }
        let mut context = Context::new(facts.0, facts.1);

        // Apply the rules until reaching a fixed point.
        let mut changed = true;
        let mut added_rules = vec![];
        let mut rules: Vec<_> = self.rules.into_iter().map(|r| (false, r)).collect();

        while changed {
            changed = false;

            for (used, rule) in &mut rules {
                // Don't try to apply rules which have already been used.
                if *used {
                    continue;
                }

                trace!("  Applying rule {:?}", rule);
                let (step_used, mut step_added) = rule.apply(&mut context)?;
                *used |= step_used;

                // There is a change if the rule was used, or if it added new rules.
                changed |= step_used;
                changed |= step_added.len() > 0;

                added_rules.append(&mut step_added);
            }

            trace!("  Applyingall rules");

            for rule in added_rules.drain(..) {
                rules.push((false, rule));
            }
        }

        trace!("  Solver exiting {:?}", context);
        for i in &mut context.inputs {
            i.reduce();
        }
        for o in &mut context.outputs {
            o.reduce();
        }
        Ok((context.inputs, context.outputs))
    }

    /// Ensures that two expressions are equal.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.equals(outputs[0].rank, inputs[1].shape[0]);
    /// solver.equals(outputs[1].rank, 3);
    /// ```
    pub fn equals<T, A, B>(&mut self, left: A, right: B) -> &mut Solver<'rules>
    where
        T: Output + Fact + 'static,
        A: IntoExp<T>,
        B: IntoExp<T>,
    {
        let items: Vec<Exp<T>> = vec![left.bex(), right.bex()];

        let rule = EqualsRule::new(items);
        self.rules.push(Box::new(rule));
        self
    }

    /// Ensures that several expressions are equal.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.equals_all(vec![
    ///     outputs[0].rank.into(),
    ///     inputs[1].shape[0].into(),
    ///     3.into(),
    /// ]);
    /// ```
    pub fn equals_all<T>(&mut self, items: Vec<Exp<T>>) -> &mut Solver<'rules>
    where
        T: Output + Fact + 'static,
    {
        let rule = EqualsRule::new(items);
        self.rules.push(Box::new(rule));
        self
    }

    /// Ensures that the sum of several expressions equals zero.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.equals_zero(vec![
    ///     outputs[0].rank.into(),
    ///     outputs[1].rank.into(),
    ///     (-1, inputs[1].shape[0]).into(),
    /// ]);
    /// ```
    pub fn equals_zero<F>(&mut self, items: Exp<F>) -> &mut Solver<'rules>
    where
        F: Fact
            + Zero
            + Add<F, Output = F>
            + Neg<Output = F>
            + Clone
            + ::std::fmt::Debug
            + Output
            + 'rules,
    {
        let rule = EqualsZeroRule(items);
        self.rules.push(Box::new(rule));
        self
    }

    /// Adds rules to the solver with a partial value.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.given(input.rank, |solver, ir|
    ///     (0..ir).map(|i| solver.equals(input.shape[ir], 0))
    /// );
    pub fn with<T, A, F>(&mut self, item: A, closure: F) -> &mut Solver<'rules>
    where
        T: Fact + Output + 'static,
        A: IntoExp<T>,
        F: Fn(&mut Solver<'rules>, T) + 'rules,
    {
        let rule = WithRule::new(item.bex(), closure);
        self.rules.push(Box::new(rule));
        self
    }

    /// Adds rules to the solver once the value of an expression is known.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.given(input.rank, |solver, ir|
    ///     (0..ir).map(|i| solver.equals(input.shape[ir], 0))
    /// );
    pub fn given<T, A, F>(&mut self, item: A, closure: F) -> &mut Solver<'rules>
    where
        T: Fact + Output + 'static,
        A: IntoExp<T>,
        F: Fn(&mut Solver<'rules>, T::Concrete) + 'rules,
    {
        let rule = GivenRule::new(item.bex(), closure);
        self.rules.push(Box::new(rule));
        self
    }

    /// Adds rules to the solver once the value of all expressions are known.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.given(input.rank, |solver, ir|
    ///     (0..ir).map(|i| solver.equals(input.shape[ir], 0))
    /// );
    pub fn given_all<T, I, A, F>(&mut self, items: I, closure: F) -> &mut Solver<'rules>
    where
        T: Fact + Output + 'static,
        A: IntoExp<T>,
        I: IntoIterator<Item = A>,
        F: Fn(&mut Solver<'rules>, Vec<T::Concrete>) + 'rules,
    {
        let rule = GivenAllRule::new(items.into_iter().map(|it| it.bex()).collect(), closure);
        self.rules.push(Box::new(rule));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use DatumType;

    fn bootstrap<'s>() -> (Solver<'s>, TensorsProxy, TensorsProxy) {
        (
            Solver::default(),
            TensorsProxy::new(tvec![0].into()),
            TensorsProxy::new(tvec![1].into()),
        )
    }

    #[test]
    #[should_panic]
    fn solver_wrong_size_1() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs.len, 2);
        solver.infer((tvec![].into(), tvec![].into())).unwrap();
    }

    #[test]
    #[should_panic]
    fn solver_wrong_size_2() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].rank, 2);
        solver.infer((tvec![].into(), tvec![].into())).unwrap();
    }

    #[test]
    fn solver_exact_size() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs.len, 1);

        let facts = solver
            .infer((tvec![TensorFact::new()].into(), tvec![].into()))
            .unwrap();
        assert_eq!(facts, (tvec![TensorFact::new()].into(), tvec![].into()));
    }

    #[test]
    fn solver_dynamic_size() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[1].datum_type, DatumType::I32);

        let facts = solver
            .infer((tvec![TensorFact::new(), TensorFact::new()], tvec![]))
            .unwrap();
        let expected = (
            tvec![
                TensorFact::new(),
                TensorFact {
                    datum_type: typefact!(DatumType::I32),
                    ..TensorFact::new()
                },
            ],
            tvec![],
        );

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_exact_rank() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].rank, 2);

        let facts = solver.infer((tvec![TensorFact::new()], tvec![])).unwrap();
        let expected = (
            tvec![TensorFact {
                shape: shapefact![_, _],
                ..TensorFact::new()
            }],
            tvec![],
        );

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_dynamic_rank() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].shape[1], 0.to_dim());

        let facts = solver.infer((tvec![TensorFact::new()], tvec![])).unwrap();
        let expected = (
            tvec![TensorFact {
                shape: shapefact![_, 0; ..],
                ..TensorFact::new()
            }],
            tvec![],
        );

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_ranks() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].rank, 3);
        solver.equals(&inputs[0].shape[0], &inputs[0].shape[1]);
        solver.equals(&inputs[0].shape[1], &inputs[0].shape[2]);
        solver.equals(&inputs[0].shape[1], 3.to_dim());

        let facts = solver.infer((tvec![TensorFact::new()], tvec![])).unwrap();
        let expected = (
            tvec![TensorFact {
                shape: shapefact![3, 3, 3],
                ..TensorFact::new()
            }],
            tvec![],
        );

        assert_eq!(facts, expected);
    }

    #[test]
    #[should_panic]
    fn solver_wrong_constant() {
        let (mut solver, _, _) = bootstrap();
        solver.equals(1, 2);
        solver.infer((tvec![], tvec![])).unwrap();
    }

    #[test]
    fn solver_right_constant() {
        let (mut solver, _, _) = bootstrap();
        solver.equals(2, 2);
        solver.infer((tvec![], tvec![])).unwrap();
    }

    #[test]
    fn solver_backward_1() {
        let (mut solver, inputs, outputs) = bootstrap();
        solver.equals(&inputs[0].shape[1], &outputs[0].shape[1]);

        let facts = solver
            .infer((tvec![TensorFact::new()], tvec![TensorFact::new()]))
            .unwrap();
        let expected = (tvec![TensorFact::new()], tvec![TensorFact::new()]);

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_backward_2() {
        let (mut solver, inputs, outputs) = bootstrap();
        solver.equals(&inputs[0].shape[1], &outputs[0].shape[1]);

        let output = TensorFact {
            shape: shapefact![_, 2, _],
            ..TensorFact::new()
        };
        let facts = solver
            .infer((tvec![TensorFact::new()], tvec![output.clone()]))
            .unwrap();
        let expected = (
            tvec![TensorFact {
                shape: shapefact![_, 2; ..],
                ..TensorFact::new()
            }],
            tvec![output.clone()],
        );

        assert_eq!(facts, expected);
    }
}
