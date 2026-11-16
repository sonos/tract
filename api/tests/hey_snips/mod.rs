const HEY_SNIPS_ASSETS: [(&str, &str); 2] = [
    ("https://tract-test-assets.tract.rs/hey-snips/v1/hey_snips.nnef.tar", "hey_snips.nnef.tar"),
    ("https://tract-test-assets.tract.rs/e2e/hey-snips-tts.wav", "hey-snips-tts.wav"),
];

fn ensure_hey_snips_assets() {
    static START: std::sync::Once = std::sync::Once::new();
    START.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
        for (url, file) in HEY_SNIPS_ASSETS {
            if std::fs::metadata(file).is_err() {
                let client = reqwest::blocking::Client::new();
                let data = client.get(url).send().unwrap().error_for_status().unwrap();
                std::fs::write(file, data.bytes().unwrap()).unwrap();
            }
        }
    });
}

fn hey_snips_speech() -> Vec<f32> {
    let wav = std::fs::read("hey-snips-tts.wav").unwrap();
    assert_eq!(&wav[36..40], b"data");
    wav[44..].as_chunks::<2>().0.iter().map(|s| i16::from_le_bytes(*s) as f32 / 32768.0).collect()
}

fn wake_ups(scores: &[f32]) -> usize {
    scores.windows(2).filter(|pair| pair[0] <= 2.0 && pair[1] > 2.0).count()
}

fn hey_snips_scores(state: &mut State, pulses: &[&[f32]]) -> anyhow::Result<Vec<f32>> {
    let mut scores = vec![];
    for pulse in pulses {
        let outputs = state.run([Tensor::from_slice(&[pulse.len()], pulse)?])?;
        scores.extend(outputs[0].as_slice::<f32>()?.chunks(2).map(|frame| frame[1] - frame[0]));
    }
    Ok(scores)
}

#[test]
fn test_hey_snips() -> anyhow::Result<()> {
    ensure_hey_snips_assets();
    let mut nnef = nnef()?;
    nnef.enable_pulse()?;
    let model = nnef.load("hey_snips.nnef.tar")?;
    let pulse = model.input_fact(0)?.dim(0)?.to_int64()? as usize;
    let delay = model.property("pulse.delay")?.as_slice::<i64>()?[0] as usize;
    let mut state = model.into_runnable()?.spawn_state()?;
    assert_eq!(state.input_count()?, 1);
    assert_eq!(state.output_count()?, 1);
    let speech = hey_snips_speech();
    let pulses: Vec<&[f32]> = speech.chunks_exact(pulse).collect();
    let (first, second) = pulses.split_at(pulses.len() / 2);
    let mut scores = hey_snips_scores(&mut state, first)?;
    let mut clone = state.clone();
    let rest = hey_snips_scores(&mut state, second)?;
    assert_eq!(hey_snips_scores(&mut clone, second)?, rest);
    scores.extend(rest);
    assert_eq!(wake_ups(&scores[delay..]), 4);
    Ok(())
}
