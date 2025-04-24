use crate::internal::*;
use crate::model::{Fact, Graph, OutletId};
use std::collections::HashMap;
use std::convert::*;
use std::fmt;

pub trait Translate<TI1, O1, TI2, O2>: fmt::Debug
where
    TI1: Fact  + Clone + 'static,
    TI2: Fact  + Clone + 'static,
    O1: fmt::Display + fmt::Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static ,
    O2: fmt::Display + fmt::Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static ,
{
    fn translate_node(
        &self,
        source: &Graph<TI1, O1>,
        node: &Node<TI1, O1>,
        target: &mut Graph<TI2, O2>,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>>;

    fn translate_model(&self, source: &Graph<TI1, O1>) -> TractResult<Graph<TI2, O2>> {
        Ok(self.translate_model_with_mappings(source)?.0)
    }

    fn translate_model_with_mappings(
        &self,
        source: &Graph<TI1, O1>,
    ) -> TractResult<(Graph<TI2, O2>, HashMap<OutletId, OutletId>)> {
        let mut target = Graph {
            symbols: source.symbols.clone(),
            .. Graph::default()
        };
        let mut mapping = HashMap::new();
        for old_id in source.eval_order()? {
            let node = source.node(old_id);
            let outlets = self
                .translate_node(source, node, &mut target, &mapping)
                .with_context(|| format!("Translating node {node} {self:?}"))?;
            for (ix, outlet) in outlets.into_iter().enumerate() {
                mapping.insert(OutletId::new(node.id, ix), outlet);
                if let Some(label) = source.outlet_label(OutletId::new(node.id, ix)) {
                    target.set_outlet_label(outlet, label.to_string())?;
                }
            }
        }
        // do not drop inputs, even if they are useless, to maintain interface
        for i in source.input_outlets()? {
            if !mapping.contains_key(i) {
                let node = source.node(i.node);
                trace!("Translate useless source {node}");
                let outlets = self
                    .translate_node(source, node, &mut target, &mapping)
                    .with_context(|| format!("Translating input {node} {self:?}"))?;
                mapping.insert(*i, outlets[0]);
            }
        }
        // maintaining order of i/o interface
        target.inputs = source.input_outlets()?.iter().map(|i| mapping[i]).collect();
        target.outputs = source.output_outlets()?.iter().map(|o| mapping[o]).collect();
        target.properties.clone_from(&source.properties);
        Ok((target, mapping))
    }
}

#[derive(Debug)]
pub struct IntoTranslator;
impl<TI1, O1, TI2, O2, EO, ETI> Translate<TI1, O1, TI2, O2> for IntoTranslator
where
    TractError: From<EO> + From<ETI>,
    TI1: Fact  + Clone + 'static,
    TI2: Fact  + for<'a> TryFrom<&'a TI1, Error = EO> + Clone + 'static,
    O1: fmt::Display + fmt::Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static ,
    O2: fmt::Display
        + for<'a> TryFrom<&'a O1, Error = ETI>
        + fmt::Debug
        + AsRef<dyn Op>
        + AsMut<dyn Op>
        + Clone
        
        + 'static,
    Graph<TI2, O2>: SpecialOps<TI2, O2>,
{
    fn translate_node(
        &self,
        source: &Graph<TI1, O1>,
        node: &Node<TI1, O1>,
        target: &mut Graph<TI2, O2>,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let node_is_input =
            (0..node.outputs.len()).all(|o| source.inputs.contains(&(node.id, o).into()));
        if node_is_input {
            (0..node.outputs.len())
                .map(|i| {
                    target.add_source(
                        if node.outputs.len() > 1 {
                            format!("{}-{}", node.name, i)
                        } else {
                            node.name.to_string()
                        },
                        TI2::try_from(&node.outputs[i].fact)?,
                    )
                })
                .collect()
        } else {
            let new_op = O2::try_from(&node.op)?;
            let facts = node
                .outputs
                .iter()
                .map(|of| Ok(TI2::try_from(&of.fact)?))
                .collect::<TractResult<TVec<_>>>()?;
            let new_id = target.add_node(node.name.clone(), new_op, facts)?;
            for (ix, o) in node.inputs.iter().enumerate() {
                target.add_edge(mapping[o], InletId::new(new_id, ix))?
            }
            Ok(node.outputs.iter().enumerate().map(|(ix, _)| OutletId::new(new_id, ix)).collect())
        }
    }
}
