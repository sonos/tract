use std::fmt;
use std::ops::{Add, Neg};

use tract_num_traits::Zero;

use crate::infer::*;

use self::super::expr::{Exp, IntoExp, Output, TExp};
use self::super::path::{get_path, set_path, Path};
use self::super::InferenceResult;

/// A structure that holds the current sets of InferenceFacts.
///
/// This is used during inference (see `Solver::infer`) to let rules compute
/// the value of expressions which involve tensor properties.
#[derive(Debug, new)]
pub struct Context {
    pub inputs: TVec<InferenceFact>,
    pub outputs: TVec<InferenceFact>,
}

impl Context {
    /// Returns the current value of the variable at the given path.
    pub fn get<T: Output>(&self, path: &Path) -> TractResult<T> {
        let value = get_path(self, &path[..])?;
        T::from_wrapped(value)
    }

    /// Tries to set the value of the variable at the given path.
    pub fn set<T: Output>(&mut self, path: &Path, value: T) -> TractResult<()> {
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
    fn apply(
        &self,
        context: &mut Context,
    ) -> TractResult<(bool, Vec<Box<dyn Rule<'rules> + 'rules>>)>;

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
struct EqualsRule<T: Output + Factoid> {
    items: Vec<Exp<T>>,
}

impl<T: Output + Factoid> EqualsRule<T> {
    /// Creates a new EqualsRule instance.
    pub fn new(items: Vec<Exp<T>>) -> EqualsRule<T> {
        EqualsRule { items }
    }
}

impl<'rules, T: Output + Factoid> Rule<'rules> for EqualsRule<T> {
    /// Tries to apply the rule to a given context.
    fn apply(
        &self,
        context: &mut Context,
    ) -> TractResult<(bool, Vec<Box<dyn Rule<'rules> + 'rules>>)> {
        let value =
            self.items.iter().try_fold(T::default(), |acc, f| acc.unify(&f.get(context)?))?;
        let mut changed = false;
        for item in &self.items {
            changed |= item.set(context, value.clone())?;
        }
        Ok((changed, vec![]))
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.items.iter().flat_map(|e| e.get_paths()).collect()
    }
}

impl<T: Output + Factoid> fmt::Debug for EqualsRule<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.items[0])?;
        for item in &self.items[1..] {
            write!(formatter, " == {item:?}")?;
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
    F: Factoid + Zero + Add<F, Output = F> + Neg<Output = F> + Clone + ::std::fmt::Debug + Output;

impl<'rules, F> Rule<'rules> for EqualsZeroRule<F>
where
    F: Factoid + Zero + Add<F, Output = F> + Neg<Output = F> + Clone + ::std::fmt::Debug + Output,
{
    /// Tries to apply the rule to a given context.
    fn apply(
        &self,
        context: &mut Context,
    ) -> TractResult<(bool, Vec<Box<dyn Rule<'rules> + 'rules>>)> {
        Ok((self.0.set(context, F::zero())?, vec![]))
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.0.get_paths()
    }
}

impl<F> fmt::Debug for EqualsZeroRule<F>
where
    F: Factoid + Zero + Add<F, Output = F> + Neg<Output = F> + Clone + ::std::fmt::Debug + Output,
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
#[allow(clippy::type_complexity)]
pub struct WithRule<'rules, T: Factoid> {
    pub item: Exp<T>,
    pub closure: Box<dyn Fn(&mut Solver<'rules>, T) -> InferenceResult + 'rules>,
}

impl<'rules, T: Output + Factoid> WithRule<'rules, T> {
    /// Creates a new GivenRule instance.
    pub fn new<F>(item: Exp<T>, closure: F) -> WithRule<'rules, T>
    where
        F: Fn(&mut Solver<'rules>, T) -> InferenceResult + 'rules,
    {
        let closure = Box::new(closure);
        WithRule { item, closure }
    }
}

impl<'rules, T: Output + Factoid> Rule<'rules> for WithRule<'rules, T> {
    /// Tries to apply the rule to a given context.
    fn apply(
        &self,
        context: &mut Context,
    ) -> TractResult<(bool, Vec<Box<dyn Rule<'rules> + 'rules>>)> {
        let value = self.item.get(context)?;
        trace!("    With rule: {:?} is {:?}", self.item, value);
        let mut solver = Solver::default();
        (self.closure)(&mut solver, value)?;
        Ok((true, solver.take_rules()))
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.item.get_paths()
    }
}

impl<T: Output + Factoid> fmt::Debug for WithRule<'_, T> {
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
#[allow(clippy::type_complexity)]
pub struct GivenRule<'rules, T: Factoid> {
    pub item: Exp<T>,
    pub closure: Box<dyn Fn(&mut Solver<'rules>, T::Concrete) -> InferenceResult + 'rules>,
}

impl<'rules, T: Output + Factoid> GivenRule<'rules, T> {
    /// Creates a new GivenRule instance.
    pub fn new<F>(item: Exp<T>, closure: F) -> GivenRule<'rules, T>
    where
        F: Fn(&mut Solver<'rules>, T::Concrete) -> InferenceResult + 'rules,
    {
        let closure = Box::new(closure);

        GivenRule { item, closure }
    }
}

impl<'rules, T: Output + Factoid> Rule<'rules> for GivenRule<'rules, T> {
    /// Tries to apply the rule to a given context.
    fn apply(
        &self,
        context: &mut Context,
    ) -> TractResult<(bool, Vec<Box<dyn Rule<'rules> + 'rules>>)> {
        let value = self.item.get(context)?;

        if let Some(value) = value.concretize() {
            trace!("    Given rule: {:?} is {:?}", self.item, value);
            // We create a new solver instance, which will be populated with
            // new rules by the code inside the closure.
            let mut solver = Solver::default();

            (self.closure)(&mut solver, value)?;

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

impl<T: Output + Factoid> fmt::Debug for GivenRule<'_, T> {
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
#[allow(clippy::type_complexity)]
pub struct GivenAllRule<'rules, T: Factoid> {
    pub items: Vec<Exp<T>>,
    pub closure: Box<dyn Fn(&mut Solver<'rules>, Vec<T::Concrete>) -> InferenceResult + 'rules>,
}

impl<'rules, T: Output + Factoid> GivenAllRule<'rules, T> {
    /// Creates a new GivenRule instance.
    pub fn new<F>(items: Vec<Exp<T>>, closure: F) -> GivenAllRule<'rules, T>
    where
        F: Fn(&mut Solver<'rules>, Vec<T::Concrete>) -> InferenceResult + 'rules,
    {
        let closure = Box::new(closure);

        GivenAllRule { items, closure }
    }
}

impl<'rules, T: Output + Factoid> Rule<'rules> for GivenAllRule<'rules, T> {
    /// Tries to apply the rule to a given context.
    fn apply(
        &self,
        context: &mut Context,
    ) -> TractResult<(bool, Vec<Box<dyn Rule<'rules> + 'rules>>)> {
        let values: Vec<T> =
            self.items.iter().map(|it| it.get(context)).collect::<TractResult<Vec<T>>>()?;
        let concrete: Vec<_> = values.iter().filter_map(|it| it.concretize()).collect();

        if concrete.len() == self.items.len() {
            trace!("    Given all rule: {:?} is {:?}", self.items, values);
            // We create a new solver instance, which will be populated with
            // new rules by the code inside the closure.
            let mut solver = Solver::default();
            (self.closure)(&mut solver, concrete)?;
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

impl<T: Output + Factoid> fmt::Debug for GivenAllRule<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GivenAllRule {:?}", self.items)
    }
}

/// A declarative constraint solver for tensors.
#[derive(Default)]
pub struct Solver<'rules> {
    // The rules used by the solver.
    pub rules: Vec<Box<dyn Rule<'rules> + 'rules>>,
}

impl<'rules> Solver<'rules> {
    /// Consumes the solver and returns the rules that it uses.
    pub fn take_rules(self) -> Vec<Box<dyn Rule<'rules> + 'rules>> {
        self.rules
    }

    /// Runs the solver on a set of InferenceFacts.
    ///
    /// This method returns:
    /// - Err(_) if a constraint couldn't be satisfied.
    /// - Ok(None) if no more information about tensors could be deduced.
    /// - Ok(Some(facts)) otherwise, with `facts` the new InferenceFacts.
    pub fn infer_facts(
        self,
        facts: (TVec<&InferenceFact>, TVec<&InferenceFact>),
    ) -> TractResult<(TVec<InferenceFact>, TVec<InferenceFact>)> {
        let mut context = Context::new(
            facts.0.into_iter().cloned().collect(),
            facts.1.into_iter().cloned().collect(),
        );

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

                trace!("  Applying rule {rule:?}");
                let (step_used, mut step_added) = rule
                    .apply(&mut context)
                    .with_context(|| format!("Applying rule {rule:?}"))?;
                *used |= step_used;

                // There is a change if the rule was used, or if it added new rules.
                changed |= step_used;
                changed |= step_added.len() > 0;

                added_rules.append(&mut step_added);
            }

            trace!("  Applying all rules");

            for rule in added_rules.drain(..) {
                rules.push((false, rule));
            }
        }

        trace!("  Solver exiting {context:?}");
        Ok((context.inputs, context.outputs))
    }

    /// Ensures that two expressions are equal.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.equals(outputs[0].rank, inputs[1].shape[0]);
    /// solver.equals(outputs[1].rank, 3);
    /// ```
    pub fn equals<T, A, B>(&mut self, left: A, right: B) -> InferenceResult
    where
        T: Output + Factoid + 'static,
        A: IntoExp<T>,
        B: IntoExp<T>,
    {
        let items: Vec<Exp<T>> = vec![left.bex(), right.bex()];

        let rule = EqualsRule::new(items);
        self.rules.push(Box::new(rule));
        Ok(())
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
    pub fn equals_all<T>(&mut self, items: Vec<Exp<T>>) -> InferenceResult
    where
        T: Output + Factoid + 'static,
    {
        let rule = EqualsRule::new(items);
        self.rules.push(Box::new(rule));
        Ok(())
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
    pub fn equals_zero<F>(&mut self, items: Exp<F>) -> InferenceResult
    where
        F: Factoid
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
        Ok(())
    }

    /// Adds rules to the solver with a partial value.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.given(input.rank, |solver, ir|
    ///     (0..ir).map(|i| solver.equals(input.shape[ir], 0))
    /// );
    /// ```
    pub fn with<T, A, F>(&mut self, item: A, closure: F) -> InferenceResult
    where
        T: Factoid + Output + 'static,
        A: IntoExp<T>,
        F: Fn(&mut Solver<'rules>, T) -> InferenceResult + 'rules,
    {
        let rule = WithRule::new(item.bex(), closure);
        self.rules.push(Box::new(rule));
        Ok(())
    }

    /// Adds rules to the solver once the value of an expression is known.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.given(input.rank, |solver, ir|
    ///     (0..ir).map(|i| solver.equals(input.shape[ir], 0))
    /// );
    /// ```
    pub fn given<T, A, F>(&mut self, item: A, closure: F) -> InferenceResult
    where
        T: Factoid + Output + 'static,
        A: IntoExp<T>,
        F: Fn(&mut Solver<'rules>, T::Concrete) -> InferenceResult + 'rules,
    {
        let rule = GivenRule::new(item.bex(), closure);
        self.rules.push(Box::new(rule));
        Ok(())
    }

    /// Adds rules to the solver once the value of all expressions are known.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.given(input.rank, |solver, ir|
    ///     (0..ir).map(|i| solver.equals(input.shape[ir], 0))
    /// );
    /// ```
    pub fn given_all<T, I, A, F>(&mut self, items: I, closure: F) -> InferenceResult
    where
        T: Factoid + Output + 'static,
        A: IntoExp<T>,
        I: IntoIterator<Item = A>,
        F: Fn(&mut Solver<'rules>, Vec<T::Concrete>) -> InferenceResult + 'rules,
    {
        let rule = GivenAllRule::new(items.into_iter().map(|it| it.bex()).collect(), closure);
        self.rules.push(Box::new(rule));
        Ok(())
    }
}

macro_rules! given_tuple {
    ($Name:ident, $name:ident, $($id:ident),*) => {
        #[allow(non_camel_case_types)]
        pub struct $Name<'rules, $($id: Factoid),*> {
            $(pub $id: Exp<$id>,)*
            pub closure: Box<dyn Fn(&mut Solver<'rules>, $($id::Concrete,)*) -> InferenceResult + 'rules>,
        }

        #[allow(non_camel_case_types)]
        impl<'rules, $($id: Factoid + Output,)*> $Name<'rules, $($id,)*> {
            pub fn new<F>($($id: Exp<$id>,)* closure: F) -> $Name<'rules, $($id,)*>
            where
                F: Fn(&mut Solver<'rules>, $($id::Concrete,)*) -> InferenceResult + 'rules,
            {
                $Name { $($id,)*
                    closure: Box::new(closure),
                }
            }
        }

        #[allow(non_camel_case_types)]
        impl<'rules, $($id: Factoid + Output,)*> Rule<'rules> for $Name<'rules, $($id,)*> {
            /// Tries to apply the rule to a given context.
            fn apply(&self, context: &mut Context) -> TractResult<(bool, Vec<Box<dyn Rule<'rules> + 'rules>>)> {
                $(
                let $id = if let Some(it) = self.$id.get(context)?.concretize() {
                    it
                } else {
                    return Ok((false, vec![]));
                };
                )*

                let mut solver = Solver::default();
                (self.closure)(&mut solver, $($id,)*)?;
                Ok((true, solver.take_rules()))
            }

            /// Returns the paths that the rule depends on.
            fn get_paths(&self) -> Vec<&Path> {
                let mut v = vec!();
                $(v.extend(self.$id.get_paths());)*
                v
            }
        }

        #[allow(non_camel_case_types)]
        impl<'s, $($id: Factoid + Output,)*> fmt::Debug for $Name<'s, $($id,)*> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "GivenRule {{ {:?} }}", ($(&self.$id),*))
            }
        }

    }
}

given_tuple!(Given2Rule, given_2, a, b);
impl<'rules> Solver<'rules> {
    pub fn given_2<T1, T2, A1, A2, F>(
        &mut self,
        item_1: A1,
        item_2: A2,
        closure: F,
    ) -> InferenceResult
    where
        A1: IntoExp<T1>,
        T1: Factoid + Output + 'static,
        A2: IntoExp<T2>,
        T2: Factoid + Output + 'static,
        F: Fn(&mut Solver<'rules>, T1::Concrete, T2::Concrete) -> InferenceResult + 'rules,
    {
        let rule = Given2Rule::new(item_1.bex(), item_2.bex(), closure);
        self.rules.push(Box::new(rule));
        Ok(())
    }
}

given_tuple!(Given3Rule, given_3, a, b, c);
impl<'rules> Solver<'rules> {
    pub fn given_3<T1, T2, T3, A1, A2, A3, F>(
        &mut self,
        item_1: A1,
        item_2: A2,
        item_3: A3,
        closure: F,
    ) -> InferenceResult
    where
        A1: IntoExp<T1>,
        T1: Factoid + Output + 'static,
        A2: IntoExp<T2>,
        T2: Factoid + Output + 'static,
        A3: IntoExp<T3>,
        T3: Factoid + Output + 'static,
        F: Fn(&mut Solver<'rules>, T1::Concrete, T2::Concrete, T3::Concrete) -> InferenceResult
            + 'rules,
    {
        let rule = Given3Rule::new(item_1.bex(), item_2.bex(), item_3.bex(), closure);
        self.rules.push(Box::new(rule));
        Ok(())
    }
}

given_tuple!(Given4Rule, given_4, a, b, c, d);
impl<'rules> Solver<'rules> {
    pub fn given_4<T1, T2, T3, T4, A1, A2, A3, A4, F>(
        &mut self,
        item_1: A1,
        item_2: A2,
        item_3: A3,
        item_4: A4,
        closure: F,
    ) -> InferenceResult
    where
        A1: IntoExp<T1>,
        T1: Factoid + Output + 'static,
        A2: IntoExp<T2>,
        T2: Factoid + Output + 'static,
        A3: IntoExp<T3>,
        T3: Factoid + Output + 'static,
        A4: IntoExp<T4>,
        T4: Factoid + Output + 'static,
        F: Fn(
                &mut Solver<'rules>,
                T1::Concrete,
                T2::Concrete,
                T3::Concrete,
                T4::Concrete,
            ) -> InferenceResult
            + 'rules,
    {
        let rule = Given4Rule::new(item_1.bex(), item_2.bex(), item_3.bex(), item_4.bex(), closure);
        self.rules.push(Box::new(rule));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bootstrap<'s>() -> (Solver<'s>, TVec<TensorProxy>, TVec<TensorProxy>) {
        (
            Solver::default(),
            tvec!(TensorProxy::new(tvec![0, 0].into())),
            tvec!(TensorProxy::new(tvec![1, 0].into())),
        )
    }

    #[test]
    #[should_panic]
    fn solver_wrong_size_1() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].rank, 2).unwrap();
        solver.infer_facts((tvec![], tvec![])).unwrap();
    }

    #[test]
    fn solver_exact_size() {
        let (solver, _, _) = bootstrap();
        let any = InferenceFact::new();

        let facts = solver.infer_facts((tvec![&any], tvec![])).unwrap();
        assert_eq!(facts, (tvec![InferenceFact::new()], tvec![]));
    }

    #[test]
    fn solver_exact_rank() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].rank, 2).unwrap();

        let any = InferenceFact::new();
        let facts = solver.infer_facts((tvec![&any], tvec![])).unwrap();
        let expected =
            (tvec![InferenceFact { shape: shapefactoid![_, _], ..InferenceFact::new() }], tvec![]);

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_dynamic_rank() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].shape[1], 0.to_dim()).unwrap();

        let any = InferenceFact::new();
        let facts = solver.infer_facts((tvec![&any], tvec![])).unwrap();
        let expected = (
            tvec![InferenceFact { shape: shapefactoid![_, 0; ..], ..InferenceFact::new() }],
            tvec![],
        );

        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_ranks() {
        let (mut solver, inputs, _) = bootstrap();
        solver.equals(&inputs[0].rank, 3).unwrap();
        solver.equals(&inputs[0].shape[0], &inputs[0].shape[1]).unwrap();
        solver.equals(&inputs[0].shape[1], &inputs[0].shape[2]).unwrap();
        solver.equals(&inputs[0].shape[1], 3.to_dim()).unwrap();

        let any = InferenceFact::new();
        let facts = solver.infer_facts((tvec![&any], tvec![])).unwrap();
        let expected = (
            tvec![InferenceFact { shape: shapefactoid![3, 3, 3], ..InferenceFact::new() }],
            tvec![],
        );

        assert_eq!(facts, expected);
    }

    #[test]
    #[should_panic]
    fn solver_wrong_constant() {
        let (mut solver, _, _) = bootstrap();
        solver.equals(1, 2).unwrap();
        solver.infer_facts((tvec![], tvec![])).unwrap();
    }

    #[test]
    fn solver_right_constant() {
        let (mut solver, _, _) = bootstrap();
        solver.equals(2, 2).unwrap();
        solver.infer_facts((tvec![], tvec![])).unwrap();
    }

    #[test]
    fn solver_backward_1() {
        let (mut solver, inputs, outputs) = bootstrap();
        solver.equals(&inputs[0].shape[1], &outputs[0].shape[1]).unwrap();

        let any = InferenceFact::new();
        let facts = solver.infer_facts((tvec![&any], tvec![&any])).unwrap();
        let expected = (
            tvec![InferenceFact::shape(shapefactoid![_,_;..])],
            tvec![InferenceFact::shape(shapefactoid![_,_;..])],
        );
        assert_eq!(facts, expected);
    }

    #[test]
    fn solver_backward_2() {
        let (mut solver, inputs, outputs) = bootstrap();
        solver.equals(&inputs[0].shape[1], &outputs[0].shape[1]).unwrap();

        let output = InferenceFact { shape: shapefactoid![_, 2, _], ..InferenceFact::new() };
        let any = InferenceFact::new();
        let facts = solver.infer_facts((tvec![&any], tvec![&output])).unwrap();
        let expected = (
            tvec![InferenceFact { shape: shapefactoid![_, 2; ..], ..InferenceFact::new() }],
            tvec![output],
        );

        assert_eq!(facts, expected);
    }
}
