mod actions;
mod audio;
mod command;
mod parser;
mod preprocessing;
mod stt;
mod vad;

use std::{
    env,
    io::{self, Write},
    path::PathBuf,
    sync::mpsc::RecvTimeoutError,
    time::Duration,
};

use anyhow::{Context, Result};

const LEVEL_WINDOW: Duration = Duration::from_millis(20);
const CALIBRATION_DURATION: Duration = Duration::from_secs(1);
const MAX_WAIT_FOR_SPEECH: Duration = Duration::from_secs(10);
const MAX_SPEECH_DURATION: Duration = Duration::from_secs(15);
const SPEECH_START_DURATION: Duration = Duration::from_millis(60);
const SPEECH_END_DURATION: Duration = Duration::from_millis(600);
const PRE_ROLL_DURATION: Duration = Duration::from_millis(300);
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

    let recording = record_utterance().context("failed to capture a speech utterance")?;

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

    let command = parser::parse(&transcript);
    match actions::execute(&command).context("failed to execute the voice command")? {
        Some(response) => println!("\nJarvis:\n{response}"),
        None => println!("\nJarvis:\nКоманда пока не поддерживается."),
    }

    Ok(())
}

fn init_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .format_target(false)
        .init();
}

fn record_utterance() -> Result<audio::Recording> {
    let expected_duration = CALIBRATION_DURATION + MAX_WAIT_FOR_SPEECH + MAX_SPEECH_DURATION;
    let session = audio::RecordingSession::start(expected_duration, LEVEL_WINDOW)
        .context("failed to start recording from the default input device")?;
    let mut detector = vad::VadDetector::new(vad::VadConfig {
        calibration_windows: windows(CALIBRATION_DURATION),
        speech_start_windows: windows(SPEECH_START_DURATION),
        speech_end_windows: windows(SPEECH_END_DURATION),
        max_wait_windows: windows(MAX_WAIT_FOR_SPEECH),
        max_speech_windows: windows(MAX_SPEECH_DURATION),
        start_margin_db: 12.0,
        end_margin_db: 6.0,
        minimum_start_level_dbfs: -35.0,
        minimum_end_level_dbfs: -40.0,
        noise_ema_alpha: 0.02,
    });
    let mut speech_start_frame = None;

    println!("Calibrating background noise for 1 second — stay quiet...");

    let speech_end_frame = loop {
        match session.recv_level_timeout(Duration::from_secs(1)) {
            Ok(level) => {
                print_level(level)?;

                match detector.observe(level) {
                    Some(vad::VadEvent::Calibrated { noise_floor_dbfs }) => {
                        println!("\nListening... noise floor: {noise_floor_dbfs:.1} dBFS");
                    }
                    Some(vad::VadEvent::SpeechStarted { at_frame }) => {
                        let pre_roll_frames =
                            frames_for_duration(session.sample_rate(), PRE_ROLL_DURATION);
                        speech_start_frame = Some(at_frame.saturating_sub(pre_roll_frames));
                        println!("\nSpeech detected.");
                    }
                    Some(vad::VadEvent::SpeechEnded { at_frame, reason }) => {
                        println!("\nSpeech ended ({reason:?}).");
                        break at_frame;
                    }
                    Some(vad::VadEvent::TimedOut) => {
                        anyhow::bail!("no speech detected within 10 seconds")
                    }
                    None => {}
                }
            }
            Err(RecvTimeoutError::Timeout) => anyhow::bail!("audio input stopped producing data"),
            Err(RecvTimeoutError::Disconnected) => anyhow::bail!("audio level stream disconnected"),
        }
    };

    let speech_start_frame = speech_start_frame.context("speech ended before it started")?;
    let recording = session.finish().context("failed to finish recording")?;

    recording.slice_frames(speech_start_frame..speech_end_frame)
}

fn windows(duration: Duration) -> usize {
    duration.as_nanos().div_ceil(LEVEL_WINDOW.as_nanos()) as usize
}

fn frames_for_duration(sample_rate: u32, duration: Duration) -> usize {
    (sample_rate as f64 * duration.as_secs_f64()).round() as usize
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
