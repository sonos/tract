use { OutputParameters, Parameters };
use tfdeploy::analyser::interface::*;
use errors::*;
use utils::*;

pub fn handle(params: Parameters, _output_params: OutputParameters) -> Result<()> {
    let model = params.tfd_model;
    let mut analyser = model.analyser(params.output_node_id)?;

    let input = params.input.clone()
        .ok_or("Exactly one of <size> or <data> must be specified.")?;

    let shape = input.shape.iter().cloned().collect::<Option<Vec<_>>>()
        .ok_or("The compare command doesn't support streaming dimensions.")?;

    // First generate random values for the inputs.
    let mut fixed_input = Vec::new();
    for &i in &params.input_node_ids {
        let data = if input.data.is_some() {
            input.data.as_ref().unwrap().clone()
        } else {
            random_tensor(shape.clone(), input.datatype)
        };

        fixed_input.push((&model.nodes()[i].name, data));
    }

    // Run unmodified graph
    let inputs = params.input_node_ids.iter().cloned().zip(fixed_input.iter().cloned().map(|pair| pair.1)).collect();
    let original_output = model.run(inputs, params.output_node_id)?;

    info!("Setting up analyser.");

    // Add hints for the input nodes.
    if let Some(input) = params.input {
        let dims = input.shape.iter()
            .map(|d| match d {
                None    => DimFact::Streamed,
                Some(i) => DimFact::Only(*i),
            })
            .collect::<Vec<_>>();

        for &i in &params.input_node_ids {
            analyser.hint(i, &TensorFact {
                datatype: typefact!(input.datatype),
                shape: ShapeFact::closed(dims.clone()),
                value: valuefact!(_),
            })?;
        }
    }

    info!("Running analyse");
    let optimized_model = analyser.to_optimized_model()?;
    info!(
        "Size of the graph after pruning: approx. {:.2?} Ko for {:?} nodes.",
        ::bincode::serialize(&analyser.nodes)?.len() as f64 * 1e-3,
        analyser.nodes.len()
    );

    // Run optimized graph
    let output_name = &optimized_model.nodes()[params.output_node_id].name;
    let inputs = fixed_input.into_iter().map(|pair| (optimized_model.node_id_by_name(pair.0).unwrap(), pair.1)).collect();
    let optimized_model_result = optimized_model.run(inputs, optimized_model.node_id_by_name(&output_name).unwrap())?;

    if original_output.len() != optimized_model_result.len() {
        bail!("Output port are different: original:{} optimized:{}",
              original_output.len(),
              optimized_model_result.len())
    }
    for (a,b) in original_output.iter().zip(optimized_model_result.iter()) {
        if !a.close_enough(b) {
            bail!("Different output {:?} and {:?}", a, b)
        }
    }
    info!("Looks good!");
    Ok(())
}

