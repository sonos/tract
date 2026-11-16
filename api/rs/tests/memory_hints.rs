use tract::prelude::*;

fn check_memory_hints(runtime_name: &str) -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("graph.nnef"),
        "version 1.0;\nextension tract_symbol T;\ngraph arithmetic(x) -> (y) {\n x = external<scalar>(shape = [T]);\n squared = mul(x, x);\n y = add(squared, x);\n}\n",
    )?;
    let model = tract::nnef()?.load(dir.path())?;
    let runtime = tract::runtime_for_name(runtime_name)?;
    assert!(runtime.prepare_with_memory_sizing_hints(model.clone(), &[("missing", 4)]).is_err());
    assert!(runtime.prepare_with_memory_sizing_hints(model.clone(), &[("T", -1)]).is_err());
    let runnable = runtime.prepare_with_memory_sizing_hints(model, &[("T", 4)])?;
    for len in [2, 7, 3] {
        let input = Tensor::from_slice(&[len], &vec![0.5f32; len])?;
        let output = runnable.run([input])?;
        assert_eq!(output[0].shape()?, vec![len]);
        assert_eq!(output[0], Tensor::from_slice(&[len], &vec![0.75f32; len])?);
    }
    Ok(())
}

#[test]
fn memory_hints_validate_symbols_without_fixing_runtime_shapes() -> anyhow::Result<()> {
    check_memory_hints("default")
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_memory_hints_allow_shapes_above_and_below_estimates() -> anyhow::Result<()> {
    check_memory_hints("metal")
}
