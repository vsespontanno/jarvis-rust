mod audio;
mod preprocessing;
mod stt;

use std::{
    env,
    io::{self, Write},
    path::PathBuf,
    sync::mpsc::RecvTimeoutError,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};

const RECORDING_DURATION: Duration = Duration::from_secs(5);
const LEVEL_WINDOW: Duration = Duration::from_millis(20);
const RAW_OUTPUT_PATH: &str = "recording.wav";
const STT_OUTPUT_PATH: &str = "recording-16khz-mono.wav";
const DEFAULT_MODEL_PATH: &str = "models/ggml-small.bin";

fn main() -> Result<()> {
    init_logging();

    let model_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));

    println!("Loading Whisper model from {}...", model_path.display());
    let transcriber = stt::WhisperTranscriber::load(&model_path)
        .context("failed to load the speech recognition model")?;
    println!("Whisper model is ready.");

    println!("Recording from the default microphone for 5 seconds...");

    let session = audio::RecordingSession::start(RECORDING_DURATION, LEVEL_WINDOW)
        .context("failed to start recording from the default input device")?;
    show_input_levels(&session, RECORDING_DURATION)?;
    let recording = session.finish().context("failed to finish recording")?;

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

fn init_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .format_target(false)
        .init();
}

fn show_input_levels(session: &audio::RecordingSession, duration: Duration) -> Result<()> {
    let deadline = Instant::now() + duration;

    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        match session.recv_level_timeout(remaining.min(Duration::from_millis(100))) {
            Ok(level) => print_level(level)?,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                anyhow::bail!("audio level stream disconnected")
            }
        }
    }

    println!();
    Ok(())
}

fn print_level(level: audio::AudioLevel) -> Result<()> {
    const METER_MIN_DBFS: f32 = -60.0;
    const METER_WIDTH: usize = 20;

    let normalized = ((level.dbfs - METER_MIN_DBFS) / -METER_MIN_DBFS).clamp(0.0, 1.0);
    let filled = (normalized * METER_WIDTH as f32).round() as usize;
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(METER_WIDTH - filled));

    print!(
        "\rLevel: {:>6.1} dBFS  RMS {:.4}  [{bar}]",
        level.dbfs, level.rms
    );
    io::stdout().flush()?;
    Ok(())
}
