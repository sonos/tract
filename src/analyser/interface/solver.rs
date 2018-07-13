use num_traits::Num;

use Result;
use analyser::types::TensorFact;
use analyser::interface::proxies::Path;
use analyser::interface::expressions::Datum;
use analyser::interface::expressions::Wrapped;
use analyser::interface::expressions::Expression;

/// A structure that holds the current value of tensor properties.
///
/// This is used during inference (see `Solver::infer`) to let rules compute
/// the value of expressions which involve tensor properties.
pub struct Context {
    dirty: bool,
}

impl Context {
    /// Creates a new Context using variables involved in the given rules.
    pub fn from(rules: &Vec<Box<Rule>>) -> Context {
        unimplemented!()
    }

    /// Sets the current value of all variables from a set of TensorFacts.
    pub fn fill(&mut self, facts: &(Vec<TensorFact>, Vec<TensorFact>)) {
        unimplemented!()
    }

    /// Dumps the current value of all variables into a set of TensorFacts.
    pub fn dump(&self) -> (Vec<TensorFact>, Vec<TensorFact>) {
        unimplemented!()
    }

    /// Returns whether the value of at least one variable has changed since
    /// the Context was created.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Returns the current value of the variables at the given path.
    pub fn get<T>(&self, path: &Vec<usize>) -> Result<Option<T>> {
        unimplemented!()
    }

    /// Tries to set the value of the variable at the given path.
    pub fn set<T>(&mut self, path: &Vec<usize>, value: T) -> Result<()> {
        unimplemented!()
    }
}

/// A rule that can be applied by the solver.
pub trait Rule {
    /// Tries to apply the rule to a given context.
    ///
    /// The method must return Ok(true) if the rule was applied successfully
    /// (meaning that the Context was mutated), or Ok(false) if the rule was
    /// not applied but didn't generate any errors.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule>>)>;
}

/// The `equals` rule.
/// It states that the given expressions must all be equal.
///
/// It can be added to the solver via the following two methods:
/// ```text
/// solver.equals(a, b);
/// solver.equals_all(vec![a, b, ...]);
/// ```
struct EqualsRule<T: Datum> {
    items: Vec<Box<Expression<Output = T>>>,
}

impl<T: Datum> EqualsRule<T> {
    /// Creates a new EqualsRule instance.
    pub fn new(items: Vec<Box<Expression<Output = T>>>) -> EqualsRule<T> {
        EqualsRule { items }
    }
}

impl<T: Datum> Rule for EqualsRule<T> {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule>>)> {
        // Find an expression which already has a value in the context.
        let mut first = None;

        for item in &self.items {
            if let Some(value) = item.get(context)? {
                first = Some(value);
                break;
            }
        }

        if let Some(value) = first {
            // All the items should have the same value.
            for item in &self.items {
                item.set(context, value)?;
            }

            Ok((true, vec![]))
        } else {
            Ok((false, vec![]))
        }
    }
}

/// The `equals_zero` rule.
/// It states that the sum of the given expressions must equal zero.
///
/// It can be added to the solver via the following method:
/// ```text
/// solver.equals_zero(vec![a, b, ...]);
/// ```
struct EqualsZeroRule<T: Datum + Num> {
    items: Vec<Box<Expression<Output = T>>>,
}

impl<T: Datum + Num> EqualsZeroRule<T> {
    /// Creates a new EqualsZeroRule instance.
    pub fn new(items: Vec<Box<Expression<Output = T>>>) -> EqualsZeroRule<T> {
        EqualsZeroRule { items }
    }
}

impl<T: Datum + Num> Rule for EqualsZeroRule<T> {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule>>)> {
        // Find all the expressions which have a value in the context.
        let mut values = vec![];
        let mut sum = T::zero();

        let mut misses = vec![];

        for item in &self.items {
            if let Some(value) = item.get(context)? {
                values.push(value);
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
        } else if sum == T::zero() {
            Ok((true, vec![]))
        } else {
            bail!("The sum of these values doesn't equal zero: {:?}.", values);
        }
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
struct GivenRule<T: Datum, E: Expression<Output = T>> {
    item: E,
    closure: Box<Fn(&mut Solver, T) -> ()>,
}

impl<T: Datum, E: Expression<Output = T>> GivenRule<T, E> {
    /// Creates a new GivenRule instance.
    pub fn new<F: 'static>(item: E, closure: F) -> GivenRule<T, E>
    where
        F: Fn(&mut Solver, T) -> ()
    {
        let closure = Box::new(closure);

        GivenRule { item, closure }
    }
}

impl<T: Datum, E: Expression<Output = T>> Rule for GivenRule<T, E> {
    /// Tries to apply the rule to a given context.
    fn apply(&self, context: &mut Context) -> Result<(bool, Vec<Box<Rule>>)> {
        if let Some(value) = self.item.get(context)? {
            // We create a new solver instance, which will be populated with
            // new rules by the code inside the closure.
            let mut solver = Solver::new();

            (self.closure)(&mut solver, value);

            Ok((true, solver.take_rules()))
        } else {
            Ok((false, vec![]))
        }
    }
}

/// A declarative constraint solver for tensors.
#[derive(new)]
pub struct Solver {
    // The rules used by the solver.
    #[new(default)]
    rules: Vec<Box<Rule>>,
}

impl Solver {
    /// Consumes the solver and returns the rules that it uses.
    pub fn take_rules(self) -> Vec<Box<Rule>> {
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
    ) -> Result<Option<(Vec<TensorFact>, Vec<TensorFact>)>> {
        // Create a Context using the variables involved in the rules and
        // fill it using the input TensorFacts.
        let mut context = Context::from(&self.rules, &facts);

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

                let (step_used, mut step_added) = rule.apply(&mut context)?;
                *used |= step_used;
                changed |= step_used;
                added_rules.append(&mut step_added);
            }
        }

        for rule in added_rules.drain(..) {
            context.update(&rule, &facts);
            rules.push((false, rule));
        }

        // Dump the output TensorFacts from the context.
        if context.is_dirty() {
            Ok(Some(context.dump()))
        } else {
            Ok(None)
        }
    }

    /// Ensures that two expressions are equal.
    ///
    /// For instance, one could write:
    /// ```text
    /// solver.equals(outputs[0].rank, inputs[1].shape[0]);
    /// solver.equals(outputs[1].rank, 3);
    /// ```
    pub fn equals<T: 'static, EA: 'static, EB: 'static, A, B>(&mut self, left: A, right: B) -> &mut Solver
    where
        T: Datum,
        EA: Expression<Output = T>,
        EB: Expression<Output = T>,
        A: Into<EA>,
        B: Into<EB>,
    {
        let items: Vec<Box<Expression<Output = T>>> =
            vec![Box::new(left.into()), Box::new(right.into())];

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
    pub fn equals_all<T: 'static>(&mut self, items: Vec<Box<Expression<Output = T>>>) -> &mut Solver
    where
        T: Datum,
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
    pub fn equals_zero<T: 'static>(&mut self, items: Vec<Box<Expression<Output = T>>>) -> &mut Solver
    where
        T: Datum + Num,
    {
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
    pub fn given<T: 'static, E: 'static, A, F: 'static>(&mut self, item: A, closure: F) -> &mut Solver
    where
        T: Datum,
        E: Expression<Output = T>,
        A: Into<E>,
        F: Fn(&mut Solver, T) -> ()
    {
        let rule = GivenRule::new(item.into(), closure);
        self.rules.push(Box::new(rule));
        self
    }
}
