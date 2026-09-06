mod audio;
mod preprocessing;
mod stt;

use std::{env, path::PathBuf, time::Duration};

use anyhow::{Context, Result};

const RECORDING_DURATION: Duration = Duration::from_secs(5);
const RAW_OUTPUT_PATH: &str = "recording.wav";
const STT_OUTPUT_PATH: &str = "recording-16khz-mono.wav";
const DEFAULT_MODEL_PATH: &str = "models/ggml-small.bin";

fn main() -> Result<()> {
    let model_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));

    println!("Loading Whisper model from {}...", model_path.display());
    let transcriber = stt::WhisperTranscriber::load(&model_path)
        .context("failed to load the speech recognition model")?;

    println!("Recording from the default microphone for 5 seconds...");

    let recording = audio::record_default_input(RECORDING_DURATION)
        .context("failed to record audio from the default input device")?;

    recording
        .write_wav(RAW_OUTPUT_PATH.as_ref())
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
        .write_wav(STT_OUTPUT_PATH.as_ref())
        .context("failed to write the preprocessed recording")?;

    println!(
        "Saved {} mono f32 samples ({} Hz) to {STT_OUTPUT_PATH}",
        stt_audio.samples.len(),
        preprocessing::STT_SAMPLE_RATE,
    );

    println!("Transcribing...");
    let transcript = transcriber
        .transcribe(&stt_audio.samples)
        .context("speech recognition failed")?;

    println!("\nTranscript:\n{transcript}");

    Ok(())
}
