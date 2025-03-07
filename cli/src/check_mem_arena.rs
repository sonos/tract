
use tract_hir::internal::*;

pub fn verify_size_and_usage(model: &TypedModel, options: &PlanOptions, path: impl AsRef<std::path::Path>) -> TractResult<()> {
    log::info!("Analyzing Metal memory schema utilization...");
    const SCHEMA_HINT_S: i64 = 1024;
    const SCHEMA_HINT_P: i64 = 0;

    const MAX_GEN_TOKENS: i64 = 2048;
    const MAX_PROMPT_TOKENS: i64 = 2048;

    const STEP_TOKENS: i64 = 16;

    let plan = SimplePlan::new_with_options(model, options)?;
    let order = plan.order_without_consts();
    let mut symbol_values = SymbolValues::default();
    let sequence_length = model
        .symbols
        .get("S")
        .context("Could not find symbol S in model")?;
    let past_sequence_length = model
        .symbols
        .get("P")
        .context("Could not find symbol P in model")?;

    symbol_values.set(&sequence_length, SCHEMA_HINT_S);
    symbol_values.set(&past_sequence_length, SCHEMA_HINT_P);

    let schema = tract_metal::memory::MetalMemSchema::build(
        model,
        order,
        &symbol_values,
    )?;

    let size_by_partition: Vec<String> = schema
        .size_by_partition()
        .iter()
        .map(|it| format!("\"{}\"", it))
        .collect();
    let mut result: String = format!(
        "{{\n\"memory_size\": \"{}\",\n\"size_by_partition\": [{}],\n\"pp\": {{",
        schema.memory_size(),
        size_by_partition.join(",\n"),
    );
    for s in (STEP_TOKENS..MAX_PROMPT_TOKENS+1).step_by(STEP_TOKENS as usize) {
        log::info!("Prompt processing: P: 0, S: {}", s);
        symbol_values.set(&sequence_length, s);
        symbol_values.set(&past_sequence_length, 0);
        if s > STEP_TOKENS {
            result += ",";
        }
        result += &format!(
            "\n\"{}\": {{\n\"peak_memory_size\": {},\n\"peak_memory_usage\": {}\n}}",
            s,
            schema.eval_peak_memory_size(&symbol_values)?,
            schema.eval_usage(&symbol_values)?,
        );
    }
    result += "\n},\n\"tg\": {";
    for p in (0..MAX_GEN_TOKENS+1).step_by(STEP_TOKENS as usize) {
        if p % STEP_TOKENS == 0 {
            log::info!("Token generation: P: {}, S: 1", p);
        }
        symbol_values.set(&sequence_length, 1);
        symbol_values.set(&past_sequence_length, p);
        if p > 0 {
            result += ",";
        }
        result += &format!(
            "\n\"{}\": {{\n\"peak_memory_size\": {},\n\"peak_memory_usage\": {}\n}}",
            p,
            schema.eval_peak_memory_size(&symbol_values)?,
            schema.eval_usage(&symbol_values)?,
        );
    }
    result += "\n}\n}\n";
    std::fs::write(path.as_ref(), result).expect("Unable to write file");

    Ok(())
}
