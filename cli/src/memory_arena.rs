
use tract_hir::internal::*;
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use tract_gpu::memory::DeviceMemSchema;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MemArenaUsage {
    arena_memory_size: i64,
    peak_memory_size: i64,
    peak_memory_usage: f32,
}

impl MemArenaUsage {
    pub fn eval_from_schema(
        schema: &DeviceMemSchema,
        symbol_values: &SymbolValues,
    ) -> TractResult<Self> {
        Ok(Self {
            arena_memory_size: schema.eval_memory_size(symbol_values)?,
            peak_memory_size: schema.eval_peak_memory_size(symbol_values)?,
            peak_memory_usage: schema.eval_usage(symbol_values)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MemArenaMetrics {
    memory_size: String,
    size_by_partition: Vec<String>,
    pp: BTreeMap<i64, MemArenaUsage>,
    tg: BTreeMap<i64, MemArenaUsage>,
    max_memory_size: i64,
    aggregate_usage: f32,
}

impl MemArenaMetrics {
    pub fn from_schema(schema: &DeviceMemSchema) -> TractResult<Self> {
        log::info!("Analyzing memory arena utilization...");
        const MAX_GEN_TOKENS: i64 = 2048;
        const MAX_PROMPT_TOKENS: i64 = 2048;

        const STEP_TOKENS: i64 = 16;
        let memory_size: String = schema.memory_size().to_string();
        let size_by_partition: Vec<String> = schema
            .size_by_partition()
            .iter()
            .map(|it| it.to_string())
            .collect();
        let symbol_scope = SymbolScope::default();
        let sequence_length =  symbol_scope.sym("S");
        let past_sequence_length = symbol_scope.sym("P");

        let mut pp = BTreeMap::new();
        let mut max_memory_size: i64 = 0;
        let mut sum_size: i64 = 0;
        let mut sum_used: i64 = 0;
        for s in (STEP_TOKENS..MAX_PROMPT_TOKENS+1).step_by(STEP_TOKENS as usize) {
            log::info!("Prompt processing: P: 0, S: {}", s);
            let symbol_values = SymbolValues::default()
                .with(&sequence_length, s)
                .with(&past_sequence_length, 0);
            let usage = MemArenaUsage::eval_from_schema(schema, &symbol_values)?;
            max_memory_size = max_memory_size.max(usage.arena_memory_size);
            sum_size += usage.arena_memory_size;
            sum_used += usage.peak_memory_size;
            pp.insert(s, usage);
        }
        let mut tg = BTreeMap::new();
        for p in (0..MAX_GEN_TOKENS+1).step_by(STEP_TOKENS as usize) {
            log::info!("Token generation: P: {}, S: 1", p);
            let symbol_values = SymbolValues::default()
                .with(&sequence_length, 1)
                .with(&past_sequence_length, p);
            let usage = MemArenaUsage::eval_from_schema(schema, &symbol_values)?;
            max_memory_size = max_memory_size.max(usage.arena_memory_size);
            sum_size += usage.arena_memory_size;
            sum_used += usage.peak_memory_size;
            tg.insert(p, usage);
        }

        let aggregate_usage = ((sum_used * 100 / sum_size.max(1)) as f32) / 100.0;
        Ok(Self { memory_size, size_by_partition, pp, tg, max_memory_size, aggregate_usage })
    }
}

pub fn dump_metrics(
    model: &TypedModel,
    options: &PlanOptions,
    path: impl AsRef<std::path::Path>
) -> TractResult<()> {
    log::info!("Analyzing Metal memory schema utilization...");
    const SCHEMA_HINT_S: i64 = 1024;
    const SCHEMA_HINT_P: i64 = 0;

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

    let schema = DeviceMemSchema::build(model, order, &symbol_values)?;

    println!("resolved_memory_size: {}", schema.eval_memory_size(&symbol_values)?);
    println!("Schema:\n{schema}");

    let metrics = MemArenaMetrics::from_schema(&schema)?;

    std::fs::write(path.as_ref(), serde_json::to_string(&metrics)?).expect("Unable to write file");

    Ok(())
}
