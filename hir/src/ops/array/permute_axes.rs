use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct PermuteAxes {
    pub axes: Option<TVec<usize>>,
}

tract_linalg::impl_dyn_hash!(PermuteAxes);

impl PermuteAxes {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TractResult<TVec<D>> {
        if let Some(ref axes) = self.axes {
            if input.len() != axes.len() {
                bail!(
                    "Op expects tensor of rank {}, input is actually of rank {}.",
                    axes.len(),
                    input.len()
                );
            }
            let mut new_shape = tvec![D::zero(); input.len()];
            for (ix, &d) in axes.iter().enumerate() {
                new_shape[ix] = input[d].clone();
            }
            Ok(new_shape)
        } else {
            let mut new_shape: TVec<D> = input.iter().cloned().collect();
            new_shape.reverse();
            Ok(new_shape)
        }
    }
}

impl Expansion for PermuteAxes {
    fn name(&self) -> Cow<str> {
        "PermuteAxes".into()
    }

    op_hir!();

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.axes)])
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape)?;
            s.equals(&outputs[0].shape, output_shape)
        })?;
        if let Some(axes) = &self.axes {
            s.equals(&outputs[0].rank, axes.len() as i32)?;
        }
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let fact = target.outlet_fact(inputs[0])?;
        let axes = if let Some(axes) = &self.axes {
            if fact.rank() != axes.len() {
                bail!(
                    "Op expects tensor of rank {}, input is actually of rank {}.",
                    axes.len(),
                    fact.rank()
                );
            }
            axes.clone()
        } else {
            (0..fact.rank()).rev().collect()
        };
        let mut wire: TVec<OutletId> = inputs.into();
        for (from, to) in perm_to_atoms(&axes) {
            wire = target.wire_node(
                format!("{}.{}_to_{}", prefix, from, to),
                AxisOp::Move(from, to),
                &wire,
            )?;
        }
        Ok(wire)
    }
}
// a, b, c is a <- b, b <- c, c <- a
fn perm_to_cycles(perm: &[usize]) -> TVec<TVec<usize>> {
    let mut cycles: TVec<TVec<usize>> = tvec!();
    let mut done = 0;
    while done < perm.len() {
        if perm[done] == done || cycles.iter().any(|c| c.contains(&done)) {
            done += 1;
            continue;
        }
        let mut cycle = tvec!();
        let mut current = done;
        loop {
            cycle.push(current);
            current = perm[current];
            if current == done {
                break;
            }
        }
        cycles.push(cycle)
    }
    cycles
}

fn is_rotation_cycle(cycle: &[usize]) -> Option<(usize, usize)> {
    if cycle.windows(2).all(|w| w[0] + 1 == w[1]) {
        Some((cycle[0], cycle[cycle.len() - 1]))
    } else if cycle[1..cycle.len()].windows(2).all(|w| w[0] - 1 == w[1])
        && cycle[cycle.len() - 1] - 1 == cycle[0]
    {
        Some((cycle[1], cycle[0]))
    } else {
        None
    }
}

pub fn perm_to_atoms(input: &[usize]) -> TVec<(usize, usize)> {
    let mut changes: TVec<(usize, usize)> = tvec!();
    'top: loop {
        let mut reached: TVec<usize> = (0..input.len()).collect();
        changes.iter().for_each(|(f, t)| {
            let axis = reached.remove(*f);
            reached.insert(*t, axis);
        });
        if &*reached == input {
            return changes;
        }
        let remaining: TVec<usize> =
            input.iter().map(|x| reached.iter().position(|y| y == x).unwrap()).collect();
        let cycles = perm_to_cycles(&remaining);
        for cycle in &cycles {
            dbg!(&cycle);
            if let Some(rot) = is_rotation_cycle(&cycle) {
                dbg!(&rot);
                changes.push(rot);
                continue 'top;
            }
        }
        changes.push((cycles[0][1], cycles[0][0]));
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_perm_to_cycles() {
        assert_eq!(perm_to_cycles(&[1, 2, 0]), tvec!(tvec!(0, 1, 2)));
        assert_eq!(perm_to_cycles(&[2, 0, 1]), tvec!(tvec!(0, 2, 1)));
        assert_eq!(perm_to_cycles(&[1, 2, 3, 0]), tvec!(tvec!(0, 1, 2, 3)));
        assert_eq!(perm_to_cycles(&[3, 0, 1, 2]), tvec!(tvec!(0, 3, 2, 1)));
        assert_eq!(perm_to_cycles(&[3, 1, 2, 0, 4]), tvec!(tvec!(0, 3)));
    }

    #[test]
    fn is_rotation() {
        assert_eq!(is_rotation_cycle(&[0, 1, 2]), Some((0, 2)));
        assert_eq!(is_rotation_cycle(&[0, 2, 1]), Some((2, 0)));
    }

    #[test]
    fn test_perm_one_rotation() {
        assert_eq!(perm_to_atoms(&[1, 2, 0, 3, 4]), tvec!((0, 2)));
    }

    #[test]
    fn test_perm_two_rotations() {
        assert_eq!(perm_to_atoms(&[1, 2, 0, 4, 3]), tvec!((0, 2), (3, 4)));
    }

    #[test]
    fn test_perm_complex() {
        assert_eq!(perm_to_atoms(&[3, 1, 2, 0, 4]), tvec!((3, 0), (1, 3)));
    }
}
