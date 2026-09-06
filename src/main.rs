mod audio;
mod preprocessing;

use std::{path::Path, time::Duration};

use anyhow::{Context, Result};

const RECORDING_DURATION: Duration = Duration::from_secs(5);
const RAW_OUTPUT_PATH: &str = "recording.wav";
const STT_OUTPUT_PATH: &str = "recording-16khz-mono.wav";

fn main() -> Result<()> {
    println!("Recording from the default microphone for 5 seconds...");

    let recording = audio::record_default_input(RECORDING_DURATION)
        .context("failed to record audio from the default input device")?;

    recording
        .write_wav(Path::new(RAW_OUTPUT_PATH))
        .context("failed to write the recording")?;

    println!(
        "Saved {} samples ({} Hz, {} channel(s)) to {RAW_OUTPUT_PATH}",
        recording.samples.len(),
        recording.sample_rate,
        recording.channels,
    );

    let stt_audio = preprocessing::prepare_for_stt(&recording)
        .context("failed to prepare the recording for speech recognition")?;
    stt_audio
        .write_wav(Path::new(STT_OUTPUT_PATH))
        .context("failed to write the preprocessed recording")?;

    println!(
        "Saved {} mono f32 samples ({} Hz) to {STT_OUTPUT_PATH}",
        stt_audio.samples.len(),
        preprocessing::STT_SAMPLE_RATE,
    );

    Ok(())
}
