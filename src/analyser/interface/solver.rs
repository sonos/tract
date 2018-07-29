use Result;
use analyser::types::{Fact, IntFact, TensorFact};
use analyser::interface::path::{Path, get_path, set_path};
use analyser::interface::expressions::Output;
use analyser::interface::expressions::Expression;
use analyser::interface::expressions::IntoExpression;

use std::fmt;

/// A structure that holds the current sets of TensorFacts.
///
/// This is used during inference (see `Solver::infer`) to let rules compute
/// the value of expressions which involve tensor properties.
#[derive(Debug, new)]
pub struct Context {
    pub inputs: Vec<TensorFact>,
    pub outputs: Vec<TensorFact>,
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
    items: Vec<Box<Expression<Output = T>>>,
}

impl<T: Output + Fact> EqualsRule<T> {
    /// Creates a new EqualsRule instance.
    pub fn new(items: Vec<Box<Expression<Output = T>>>) -> EqualsRule<T> {
        EqualsRule { items }
    }
}

impl<'rules, T: Output + Fact> Rule<'rules> for EqualsRule<T> {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule<'rules> + 'rules>>)> {
        if self.items.len() < 1 {
            return Ok((false, vec![]));
        }

        // Unify the value of all the expressions into one.
        let mut value: T = Default::default();
        for item in &self.items {
            value = value.unify(&item.get(context)?)?;
        }

        if value != Default::default() {
            // Set all the values to this unified one.
            for item in &self.items {
                item.set(context, value.clone())?;
            }

            Ok((true, vec![]))
        } else {
            Ok((false, vec![]))
        }
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
/// It states that the sum of the given expressions must equal zero.
///
/// It can be added to the solver via the following method:
/// ```text
/// solver.equals_zero(vec![a, b, ...]);
/// ```
struct EqualsZeroRule {
    items: Vec<Box<Expression<Output = IntFact>>>,
}

impl EqualsZeroRule {
    /// Creates a new EqualsZeroRule instance.
    pub fn new(items: Vec<Box<Expression<Output = IntFact>>>) -> EqualsZeroRule {
        EqualsZeroRule { items }
    }
}

impl<'rules> Rule<'rules> for EqualsZeroRule {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule<'rules> + 'rules>>)> {
        // Find all the expressions which have a value in the context.
        let mut values = vec![];
        let mut sum: IntFact = 0usize.into();

        let mut misses = vec![];

        for item in &self.items {
            let value = item.get(context)?;

            if value.is_concrete() {
                values.push(value.clone());
                sum = sum + value;
            } else {
                misses.push(item);
            }
        }

        if misses.len() > 1 {
            Ok((false, vec![]))
        } else if misses.len() == 1 {
            misses[0].set(context, sum)?;
            Ok((true, vec![]))
        } else if sum == 0usize.into() {
            Ok((true, vec![]))
        } else {
            bail!("The sum of these values doesn't equal zero: {:?}.", values);
        }
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.items.iter().flat_map(|e| e.get_paths()).collect()
    }
}

impl fmt::Debug for EqualsZeroRule {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.items[0])?;
        for item in &self.items[1..] {
            write!(formatter, " + {:?}", item)?;
        }
        write!(formatter, " == 0")
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
pub struct GivenRule<'rules, T: Output + Fact, E: Expression<Output = T>, C: Output> {
    pub item: E,
    pub closure: Box<Fn(&mut Solver<'rules>, C) + 'rules>,
}

impl<'rules, T: Output + Fact, E: Expression<Output = T>, C: Output> GivenRule<'rules, T, E, C> {
    /// Creates a new GivenRule instance.
    pub fn new<F>(item: E, closure: F) -> GivenRule<'rules, T, E, C>
    where
        F: Fn(&mut Solver<'rules>, C) + 'rules
    {
        let closure = Box::new(closure);

        GivenRule { item, closure }
    }
}

impl<'rules, T: Output + Fact, E: Expression<Output = T>, C: Output> Rule<'rules> for GivenRule<'rules, T, E, C> {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule<'rules> + 'rules>>)> {
        // When calling `self.item.get(context)?`, we get a value of type T.
        // However, the developer might have wanted to explicitely convert
        // this value into a value of type C (using type annotations on the
        // closure parameters), so we need to perform that conversion.
        //
        // Thankfully, because both T and C implement Output, the conversion
        // is as simple as wrapping and un-wrapping the value.
        let value = C::from_wrapped(self.item.get(context)?.wrap());

        if let Ok(value) = value {
            // We create a new solver instance, which will be populated with
            // new rules by the code inside the closure.
            let mut solver = Solver::new();

            (self.closure)(&mut solver, value);

            Ok((true, solver.take_rules()))
        } else {
            Ok((false, vec![]))
        }
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.item.get_paths()
    }
}

impl<'s, T: Output + Fact, E: Expression<Output = T>, C: Output> fmt::Debug for GivenRule<'s, T, E, C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GivenRule {{ {:?} }}", self.item)
    }
}

/// A declarative constraint solver for tensors.
#[derive(new)]
pub struct Solver<'rules> {
    // The rules used by the solver.
    #[new(default)]
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
        facts: (Vec<TensorFact>, Vec<TensorFact>),
    ) -> Result<(Vec<TensorFact>, Vec<TensorFact>)> {
        let mut context = Context::new(facts.0, facts.1);

        // Apply the rules until reaching a fixed point.
        let mut changed = true;
        let mut added_rules = vec![];
        let mut rules: Vec<_> = self.rules.into_iter()
            .map(|r| (false, r))
            .collect();

        while changed {
            changed = false;

            for (used, rule) in &mut rules {
                // Don't try to apply rules which have already been used.
                if *used {
                    continue;
                }

                debug!("Applying rule {:?}", rule);
                let (step_used, mut step_added) = rule.apply(&mut context)?;
                *used |= step_used;

                // There is a change if the rule was used, or if it added new rules.
                changed |= step_used;
                changed |= step_added.len() > 0;

                added_rules.append(&mut step_added);
            }

            for rule in added_rules.drain(..) {
                rules.push((false, rule));
            }
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
    pub fn equals<T, EA , EB, A, B>(&mut self, left: A, right: B) -> &mut Solver<'rules>
    where
        T: Output + Fact + 'static,
        EA: Expression<Output = T> + 'static,
        EB: Expression<Output = T> + 'static,
        A: IntoExpression<EA>,
        B: IntoExpression<EB>,
    {
        let items: Vec<Box<Expression<Output = T>>> = wrap![left, right];

        let rule = EqualsRule::new(items);
        self.rules.push(Box::new(rule));
        self
    }

    /// Ensures that an several expressions are equal.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.equals_all(vec![
    ///     outputs[0].rank.into(),
    ///     inputs[1].shape[0].into(),
    ///     3.into(),
    /// ]);
    /// ```
    pub fn equals_all<T>(&mut self, items: Vec<Box<Expression<Output = T>>>) -> &mut Solver<'rules>
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
    pub fn equals_zero(&mut self, items: Vec<Box<Expression<Output = IntFact>>>) -> &mut Solver<'rules> {
        let rule = EqualsZeroRule::new(items);
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
    pub fn given<T, E, C, A, F>(&mut self, item: A, closure: F) -> &mut Solver<'rules>
    where
        T: Output + Fact + 'static,
        E: Expression<Output = T> + 'static,
        C: Output + 'static,
        A: IntoExpression<E>,
        F: Fn(&mut Solver<'rules>, C) + 'rules
    {
        let rule = GivenRule::new(item.into_expr(), closure);
        self.rules.push(Box::new(rule));
        self
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use analyser::interface::TensorsProxy;
    use tfpb::types::DataType;

    fn bootstrap<'s>() -> (Solver<'s>, TensorsProxy, TensorsProxy) {
        (Solver::new(),
         TensorsProxy::new(vec![0].into()),
         TensorsProxy::new(vec![1].into()))
    }

    #[test]
    #[should_panic]
    fn solver_wrong_size_1() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs.len, 2);
        solver.infer((vec![].into(), vec![].into())).unwrap();
    }

    #[test]
    #[should_panic]
    fn solver_wrong_size_2() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].rank, 2);
        solver.infer((vec![].into(), vec![].into())).unwrap();
    }

    #[test]
    fn solver_exact_size() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs.len, 1);

        let facts = solver.infer((vec![TensorFact::new()].into(), vec![].into())).unwrap();
        assert_eq!(facts, (vec![TensorFact::new()].into(), vec![].into()));
    }

    #[test]
    fn solver_dynamic_size() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[1].datatype, DataType::DT_INT32);

        let facts = solver.infer((vec![TensorFact::new(), TensorFact::new()], vec![])).unwrap();
        let expected = (
            vec![
                TensorFact::new(),
                TensorFact {datatype: typefact!(DataType::DT_INT32), ..TensorFact::new()}
            ],
            vec![]
        );

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_exact_rank() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].rank, 2);

        let facts = solver.infer((vec![TensorFact::new()], vec![])).unwrap();
        let expected = (
            vec![TensorFact {shape: shapefact![_, _], ..TensorFact::new()}],
            vec![]
        );

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_dynamic_rank() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].shape[1], 0);

        let facts = solver.infer((vec![TensorFact::new()], vec![])).unwrap();
        let expected = (
            vec![TensorFact {shape: shapefact![_, 0; ..], ..TensorFact::new()}],
            vec![]
        );

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_ranks() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].rank, 3);
        solver.equals(&inputs[0].shape[0], &inputs[0].shape[1]);
        solver.equals(&inputs[0].shape[1], &inputs[0].shape[2]);
        solver.equals(&inputs[0].shape[1], 3);

        let facts = solver.infer((vec![TensorFact::new()], vec![])).unwrap();
        let expected = (
            vec![TensorFact {shape: shapefact![3, 3, 3], ..TensorFact::new()}],
            vec![]
        );

        assert_eq!(facts, expected);
    }

    #[test]
    #[should_panic]
    fn solver_wrong_constant() {
        let (mut solver, _, _) = bootstrap();
        solver.equals(1, 2);
        solver.infer((vec![], vec![])).unwrap();
    }

    #[test]
    fn solver_right_constant() {
        let (mut solver, _, _) = bootstrap();
        solver.equals(2, 2);
        solver.infer((vec![], vec![])).unwrap();
    }

    #[test]
    fn solver_backward_1() {
        let (mut solver, inputs, outputs) = bootstrap();
        solver.equals(&inputs[0].shape[1], &outputs[0].shape[1]);

        let facts = solver.infer((vec![TensorFact::new()], vec![TensorFact::new()])).unwrap();
        let expected = (
            vec![TensorFact::new()],
            vec![TensorFact::new()]
        );

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_backward_2() {
        let (mut solver, inputs, outputs) = bootstrap();
        solver.equals(&inputs[0].shape[1], &outputs[0].shape[1]);

        let output = TensorFact { shape: shapefact![_, 2, _], ..TensorFact::new() };
        let facts = solver.infer((vec![TensorFact::new()], vec![output.clone()])).unwrap();
        let expected = (
            vec![TensorFact { shape: shapefact![_, 2; ..], ..TensorFact::new() }],
            vec![output.clone()]
        );

        assert_eq!(facts, expected);
    }
}
