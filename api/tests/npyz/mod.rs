use half::f16;

fn sample() -> Vec<(&'static str, Tensor)> {
    let halves: Vec<f16> = [0.5f32, -1.0, 2.0].iter().map(|x| f16::from_f32(*x)).collect();
    vec![
        ("input", Tensor::from_slice(&[2, 2], &[1f32, 2.0, 3.0, 4.0]).unwrap()),
        ("turn_0/half", Tensor::from_slice(&[3], &halves).unwrap()),
        ("mask", Tensor::from_slice(&[2], &[true, false]).unwrap()),
        ("scalar", Tensor::from_slice(&[], &[-7i64]).unwrap()),
    ]
}

#[test]
fn npy_round_trips() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    for (name, tensor) in sample() {
        let path = dir.path().join(format!("{}.npy", name.replace('/', "_")));
        tensor.save_npy(&path)?;
        assert_eq!(Tensor::load_npy(&path)?, tensor);
        let buffer = tensor.to_npy_buffer()?;
        assert_eq!(buffer, std::fs::read(&path)?);
        assert_eq!(Tensor::load_npy_buffer(&buffer)?, tensor);
    }
    Ok(())
}

#[test]
fn npy_reads_numpy_output() -> anyhow::Result<()> {
    // numpy.save(f, numpy.array([1.0, 2.0], dtype="<f4"))
    let mut file = b"\x93NUMPY\x01\x00\x76\x00".to_vec();
    file.extend(format!("{:<117}\n", "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }").as_bytes());
    file.extend([0, 0, 0x80, 0x3f, 0, 0, 0, 0x40]);
    let tensor = Tensor::load_npy_buffer(&file)?;
    assert_eq!(tensor, Tensor::from_slice(&[2], &[1f32, 2.0])?);
    assert_eq!(tensor.to_npy_buffer()?, file);
    Ok(())
}

#[test]
fn npy_rejects_garbage() {
    assert!(Tensor::load_npy_buffer(b"not a npy file").is_err());
    assert!(Tensor::load_npy("/definitely/not/here.npy").is_err());
}

#[test]
fn npz_round_trips() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let tensors = sample();
    for compress in [false, true] {
        let path = dir.path().join(format!("io-{compress}.npz"));
        npz::save(&path, tensors.iter().map(|(n, t)| (n, t)), compress)?;
        let buffer = npz::to_buffer(tensors.iter().map(|(n, t)| (n, t)), compress)?;
        let expected_names: Vec<String> = tensors.iter().map(|(n, _)| n.to_string()).collect();

        assert_eq!(npz::names(&path)?, expected_names);
        assert_eq!(npz::names_buffer(&buffer)?, expected_names);
        for (name, tensor) in &tensors {
            assert_eq!(&npz::get(&path, name)?, tensor);
            assert_eq!(&npz::get_buffer(&buffer, name)?, tensor);
            assert_eq!(&npz::get(&path, &format!("{name}.npy"))?, tensor);
        }
        for loaded in [npz::load(&path)?, npz::load_buffer(&buffer)?] {
            assert_eq!(loaded.len(), tensors.len());
            for ((name, tensor), (expected_name, expected)) in loaded.iter().zip(&tensors) {
                assert_eq!(name, expected_name);
                assert_eq!(tensor, expected);
            }
        }
        assert!(npz::get(&path, "missing").is_err());
        assert!(npz::get_buffer(&buffer, "missing").is_err());
    }
    Ok(())
}

#[test]
fn npz_empty_and_garbage() -> anyhow::Result<()> {
    let buffer = npz::to_buffer(std::iter::empty::<(&str, &Tensor)>(), true)?;
    assert!(npz::names_buffer(&buffer)?.is_empty());
    assert!(npz::load_buffer(&buffer)?.is_empty());
    assert!(npz::names_buffer(b"not a zip").is_err());
    Ok(())
}
