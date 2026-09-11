mod actions;
mod audio;
mod command;
mod dataset;
mod parser;
mod preprocessing;
mod stt;
mod vad;

use std::{
    env,
    ffi::OsStr,
    io::{self, Write},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, RecvTimeoutError},
    },
    time::Duration,
};

use anyhow::{Context, Result};

const LEVEL_WINDOW: Duration = Duration::from_millis(20);
const CALIBRATION_DURATION: Duration = Duration::from_secs(1);
const MAX_WAIT_FOR_SPEECH: Duration = Duration::from_secs(10);
const MAX_SPEECH_DURATION: Duration = Duration::from_secs(15);
const SPEECH_START_DURATION: Duration = Duration::from_millis(60);
const SPEECH_END_DURATION: Duration = Duration::from_millis(600);
const MINIMUM_SPEECH_DURATION: Duration = Duration::from_millis(140);
const PRE_ROLL_DURATION: Duration = Duration::from_millis(300);
const COLLECTION_START_MARGIN_DB: f32 = 6.0;
const COLLECTION_END_MARGIN_DB: f32 = 2.0;
const DEFAULT_MODEL_PATH: &str = "models/ggml-small.bin";
const DATASET_PATH: &str = "data";
const DATASET_REPORT_FLAG: &str = "--dataset-report";
const DATASET_EXPORT_FLAG: &str = "--dataset-export";
const DATASET_REVIEW_FLAG: &str = "--dataset-review";
const DATASET_RELABEL_FLAG: &str = "--dataset-relabel";
const DATASET_COLLECT_FLAG: &str = "--dataset-collect";
const DEFAULT_COLLECTION_PLAN_PATH: &str = "collection-plans/session-01.json";

struct CapturedUtterance {
    recording: audio::Recording,
    vad: dataset::VadTelemetry,
}

#[derive(Clone, Copy)]
enum ProcessingMode<'a> {
    Natural,
    Prompted {
        campaign: &'a str,
        prompt: &'a dataset::CollectionPrompt,
        labels: &'a dataset::LabelStore,
    },
}

fn main() -> Result<()> {
    let first_argument = env::args_os().nth(1);
    if first_argument.as_deref() == Some(OsStr::new(DATASET_REPORT_FLAG)) {
        let events_path = env::args_os()
            .nth(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DATASET_PATH).join("events.jsonl"));
        return dataset::print_report(&events_path);
    }
    if first_argument.as_deref() == Some(OsStr::new(DATASET_EXPORT_FLAG)) {
        let output_path = env::args_os()
            .nth(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DATASET_PATH).join("splits"));
        return dataset::export(PathBuf::from(DATASET_PATH).as_path(), &output_path);
    }
    if first_argument.as_deref() == Some(OsStr::new(DATASET_REVIEW_FLAG)) {
        let dataset_path = env::args_os()
            .nth(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DATASET_PATH));
        return dataset::review(&dataset_path);
    }
    if first_argument.as_deref() == Some(OsStr::new(DATASET_RELABEL_FLAG)) {
        let event_id = env::args_os()
            .nth(2)
            .context("usage: --dataset-relabel <event-id> [dataset-root]")?;
        let dataset_path = env::args_os()
            .nth(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DATASET_PATH));
        return dataset::relabel(&dataset_path, &event_id.to_string_lossy());
    }

    let collection_plan_path =
        (first_argument.as_deref() == Some(OsStr::new(DATASET_COLLECT_FLAG))).then(|| {
            env::args_os()
                .nth(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_COLLECTION_PLAN_PATH))
        });
    let collection_plan = collection_plan_path
        .as_deref()
        .map(dataset::CollectionPlan::load)
        .transpose()?;

    init_logging();
    let running = install_shutdown_handler()?;

    let model_path = if collection_plan_path.is_some() {
        PathBuf::from(DEFAULT_MODEL_PATH)
    } else {
        first_argument
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
    };

    println!("Loading Whisper model from {}...", model_path.display());
    let transcriber = stt::WhisperTranscriber::load(&model_path)
        .context("failed to load the speech recognition model")?;
    println!("Whisper model is ready.");
    let vad_config = if collection_plan.is_some() {
        collection_vad_config()
    } else {
        vad_config()
    };
    let provenance = dataset::Provenance {
        jarvis_version: env!("CARGO_PKG_VERSION").to_owned(),
        session_id: dataset::Provenance::new_session_id(),
        whisper_model: model_path.display().to_string(),
        parser_version: parser::VERSION,
        vad_config: dataset::VadConfigMetadata {
            level_window_ms: LEVEL_WINDOW.as_millis() as u64,
            calibration_ms: CALIBRATION_DURATION.as_millis() as u64,
            speech_start_ms: SPEECH_START_DURATION.as_millis() as u64,
            speech_end_ms: SPEECH_END_DURATION.as_millis() as u64,
            min_speech_duration_ms: MINIMUM_SPEECH_DURATION.as_millis() as u64,
            pre_roll_ms: PRE_ROLL_DURATION.as_millis() as u64,
            start_margin_db: vad_config.start_margin_db,
            end_margin_db: vad_config.end_margin_db,
            minimum_start_level_dbfs: vad_config.minimum_start_level_dbfs,
            minimum_end_level_dbfs: vad_config.minimum_end_level_dbfs,
        },
    };
    let dataset = dataset::DatasetStore::open(DATASET_PATH, provenance)
        .context("failed to initialize the local dataset")?;
    let expected_capture_duration =
        CALIBRATION_DURATION + MAX_WAIT_FOR_SPEECH + MAX_SPEECH_DURATION;
    let audio_input = audio::RecordingSession::start(expected_capture_duration, LEVEL_WINDOW)
        .context("failed to open the default microphone")?;
    let mut detector = vad::VadDetector::new(vad_config);
    let (timers, timer_events) = actions::timer_channel();

    if let Some(plan) = collection_plan {
        let labels = dataset::LabelStore::open(PathBuf::from(DATASET_PATH).as_path())?;
        return run_collection(
            &plan,
            &transcriber,
            &dataset,
            &labels,
            &audio_input,
            &mut detector,
            &timers,
            &timer_events,
            &running,
        );
    }

    println!("Jarvis is running. Press Ctrl+C to stop.");

    while running.load(Ordering::Relaxed) {
        if let Err(error) = process_command(
            &transcriber,
            &dataset,
            &audio_input,
            &mut detector,
            &timers,
            &timer_events,
            &running,
            ProcessingMode::Natural,
        ) && running.load(Ordering::Relaxed)
        {
            eprintln!("\nCould not process this command: {error:#}");
        }
    }

    println!("\nJarvis stopped.");
    Ok(())
}

fn install_shutdown_handler() -> Result<Arc<AtomicBool>> {
    let running = Arc::new(AtomicBool::new(true));
    let signal_flag = Arc::clone(&running);
    ctrlc::set_handler(move || signal_flag.store(false, Ordering::Relaxed))
        .context("failed to install the Ctrl+C handler")?;

    Ok(running)
}

#[allow(clippy::too_many_arguments)]
fn process_command(
    transcriber: &stt::WhisperTranscriber,
    dataset: &dataset::DatasetStore,
    audio_input: &audio::RecordingSession,
    detector: &mut vad::VadDetector,
    timers: &actions::TimerScheduler,
    timer_events: &Receiver<actions::TimerElapsed>,
    running: &AtomicBool,
    mode: ProcessingMode<'_>,
) -> Result<bool> {
    let prompted = matches!(mode, ProcessingMode::Prompted { .. });
    let Some(captured) = record_utterance(audio_input, detector, timer_events, running, prompted)
        .context("failed to capture a speech utterance")?
    else {
        return Ok(false);
    };

    let sample = dataset.new_sample();
    let mut record = dataset.new_record(&sample, &captured.recording);
    record.vad = Some(captured.vad);
    if let ProcessingMode::Prompted {
        campaign, prompt, ..
    } = mode
    {
        record.collection = Some(dataset::CollectionMetadata {
            source: dataset::CollectionSource::Prompted,
            campaign: campaign.to_owned(),
            prompt_id: prompt.id.clone(),
            expected_transcript: prompt.transcript.clone(),
            expected_intent: prompt.intent.clone(),
        });
    }

    if !has_minimum_speech(record.vad.as_ref().expect("VAD telemetry is set")) {
        let speech_duration_ms = record
            .vad
            .as_ref()
            .and_then(|vad| vad.speech_windows)
            .unwrap_or(0) as u128
            * LEVEL_WINDOW.as_millis();
        if prompted {
            println!(
                "Short prompted audio candidate ({speech_duration_ms} ms of detected speech; minimum is {} ms).",
                MINIMUM_SPEECH_DURATION.as_millis(),
            );
            dataset
                .save_audio(&sample, &captured.recording)
                .context("failed to save prompted false-negative audio")?;
            println!(
                "Saved prompted false-negative audio to {}.",
                sample.audio_path
            );
        } else {
            println!(
                "Discarded short audio candidate ({speech_duration_ms} ms of speech; minimum is {} ms).",
                MINIMUM_SPEECH_DURATION.as_millis(),
            );
            record.audio_path = None;
        }
        record.execution_result = dataset::ExecutionResult {
            status: dataset::ExecutionStatus::Rejected,
            response: None,
            error: None,
        };
        dataset
            .append(&record)
            .context("failed to append the rejected detection event")?;
        append_prompt_label(mode, &record.id)?;
        return Ok(true);
    }

    let result = process_recording(
        transcriber,
        dataset,
        &sample,
        &captured.recording,
        &mut record,
        timers,
        matches!(mode, ProcessingMode::Natural),
    );
    if let Err(error) = &result {
        record.processing_error = Some(format!("{error:#}"));
    }
    dataset
        .append(&record)
        .context("failed to append the dataset event")?;
    append_prompt_label(mode, &record.id)?;

    result.map(|_| true)
}

fn process_recording(
    transcriber: &stt::WhisperTranscriber,
    dataset: &dataset::DatasetStore,
    sample: &dataset::SampleDescriptor,
    recording: &audio::Recording,
    record: &mut dataset::DatasetRecord,
    timers: &actions::TimerScheduler,
    execute_actions: bool,
) -> Result<()> {
    dataset.save_audio(sample, recording)?;

    println!(
        "Saved {} samples ({} Hz, {} channel(s)) to {}",
        recording.samples.len(),
        recording.sample_rate,
        recording.channels,
        sample.audio_path,
    );

    let stt_audio = preprocessing::prepare_for_stt(recording)
        .context("failed to prepare the recording for speech recognition")?;

    println!("Transcribing...");
    let transcript = transcriber
        .transcribe(&stt_audio.samples)
        .context("speech recognition failed")?;
    record.transcript = Some(transcript.clone());

    println!("\nTranscript:\n{transcript}");

    let command = parser::parse(&transcript);
    record.prediction = Some(dataset::IntentPrediction {
        intent: command.intent_name().to_owned(),
        slots: command.slots(),
    });

    if command == command::Command::NoSpeech {
        println!("\nIgnored Whisper non-speech annotation.");
        record.execution_result = dataset::ExecutionResult {
            status: dataset::ExecutionStatus::Rejected,
            response: None,
            error: None,
        };
        return Ok(());
    }

    if !execute_actions {
        println!(
            "\nPredicted intent: {}\nCollection mode: action not executed.",
            command.intent_name()
        );
        return Ok(());
    }

    match actions::execute(&command, timers) {
        Ok(Some(response)) => {
            println!("\nJarvis:\n{response}");
            record.execution_result = dataset::ExecutionResult {
                status: dataset::ExecutionStatus::Succeeded,
                response: Some(response),
                error: None,
            };
        }
        Ok(None) => {
            let response = "Команда пока не поддерживается.".to_owned();
            println!("\nJarvis:\n{response}");
            record.execution_result = dataset::ExecutionResult {
                status: dataset::ExecutionStatus::Unsupported,
                response: Some(response),
                error: None,
            };
        }
        Err(error) => {
            record.execution_result = dataset::ExecutionResult {
                status: dataset::ExecutionStatus::Failed,
                response: None,
                error: Some(format!("{error:#}")),
            };
            return Err(error).context("failed to execute the voice command");
        }
    }

    Ok(())
}

fn append_prompt_label(mode: ProcessingMode<'_>, event_id: &str) -> Result<()> {
    let ProcessingMode::Prompted { prompt, labels, .. } = mode else {
        return Ok(());
    };
    labels
        .append(&dataset::DatasetLabel::new(
            event_id.to_owned(),
            true,
            Some(prompt.transcript.clone()),
            prompt.intent.clone(),
            None,
        ))
        .context("failed to append the prompted ground-truth label")
}

#[allow(clippy::too_many_arguments)]
fn run_collection(
    plan: &dataset::CollectionPlan,
    transcriber: &stt::WhisperTranscriber,
    dataset: &dataset::DatasetStore,
    labels: &dataset::LabelStore,
    audio_input: &audio::RecordingSession,
    detector: &mut vad::VadDetector,
    timers: &actions::TimerScheduler,
    timer_events: &Receiver<actions::TimerElapsed>,
    running: &AtomicBool,
) -> Result<()> {
    let completed_for_campaign =
        dataset::completed_prompt_ids(PathBuf::from(DATASET_PATH).as_path(), &plan.campaign)?;
    let completed = plan
        .prompts
        .iter()
        .filter(|prompt| completed_for_campaign.contains(&prompt.id))
        .map(|prompt| prompt.id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    println!(
        "Controlled dataset collection: {} ({} prompts).",
        plan.campaign,
        plan.prompts.len()
    );
    if !completed.is_empty() {
        println!(
            "Resuming campaign: {} prompt(s) already collected and labeled.",
            completed.len()
        );
    }
    println!(
        "Collection VAD profile: start +{COLLECTION_START_MARGIN_DB:.0} dB, end +{COLLECTION_END_MARGIN_DB:.0} dB."
    );
    println!("Voice actions are disabled. Press Ctrl+C to stop.\n");

    let mut collected = 0;
    let mut skipped = 0;
    for (index, prompt) in plan.prompts.iter().enumerate() {
        if completed.contains(&prompt.id) {
            continue;
        }
        if !running.load(Ordering::Relaxed) {
            break;
        }

        loop {
            println!(
                "[{}/{}] {}\nSay: \"{}\"",
                index + 1,
                plan.prompts.len(),
                prompt.intent,
                prompt.transcript
            );
            print!("Press Enter when ready, s=skip, q=quit: ");
            io::stdout().flush()?;
            let mut answer = String::new();
            if io::stdin().read_line(&mut answer)? == 0 || answer.trim().eq_ignore_ascii_case("q") {
                print_collection_summary(
                    collected,
                    skipped,
                    plan.prompts.len() - completed.len() - collected - skipped,
                );
                return Ok(());
            }
            if answer.trim().eq_ignore_ascii_case("s") {
                skipped += 1;
                println!();
                break;
            }
            if !answer.trim().is_empty() {
                println!("Use Enter, s, or q.\n");
                continue;
            }

            match process_command(
                transcriber,
                dataset,
                audio_input,
                detector,
                timers,
                timer_events,
                running,
                ProcessingMode::Prompted {
                    campaign: &plan.campaign,
                    prompt,
                    labels,
                },
            ) {
                Ok(true) => {
                    collected += 1;
                    println!("Prompt saved and labeled.\n");
                    break;
                }
                Ok(false) if running.load(Ordering::Relaxed) => {
                    println!("Nothing captured; retrying this prompt.\n");
                }
                Ok(false) => break,
                Err(error) => {
                    eprintln!("Could not process this prompt: {error:#}\n");
                    break;
                }
            }
        }
    }

    print_collection_summary(
        collected,
        skipped,
        plan.prompts.len() - completed.len() - collected - skipped,
    );
    Ok(())
}

fn print_collection_summary(collected: usize, skipped: usize, remaining: usize) {
    println!("Collection summary:");
    println!("  Collected: {collected}");
    println!("  Skipped: {skipped}");
    println!("  Remaining: {remaining}");
}

fn has_minimum_speech(telemetry: &dataset::VadTelemetry) -> bool {
    telemetry
        .speech_windows
        .is_some_and(|speech_windows| speech_windows >= windows(MINIMUM_SPEECH_DURATION))
}

fn init_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .format_target(false)
        .init();
}

fn vad_config() -> vad::VadConfig {
    vad::VadConfig {
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
    }
}

fn collection_vad_config() -> vad::VadConfig {
    vad::VadConfig {
        start_margin_db: COLLECTION_START_MARGIN_DB,
        end_margin_db: COLLECTION_END_MARGIN_DB,
        ..vad_config()
    }
}

fn record_utterance(
    session: &audio::RecordingSession,
    detector: &mut vad::VadDetector,
    timer_events: &Receiver<actions::TimerElapsed>,
    running: &AtomicBool,
    preserve_missed_capture: bool,
) -> Result<Option<CapturedUtterance>> {
    let needs_calibration = !detector.is_calibrated();
    detector.begin_utterance();
    session.begin_capture()?;
    let mut speech_start_frame = None;

    if needs_calibration {
        println!("Calibrating background noise for 1 second — stay quiet...");
    } else {
        let noise_floor = detector
            .noise_floor_dbfs()
            .expect("calibrated VAD has a noise floor");
        print_vad_thresholds(detector, noise_floor);
    }

    let (speech_end_frame, vad_metrics) = loop {
        if let Ok(timer) = timer_events.try_recv() {
            session.cancel_capture()?;
            if let Err(error) = actions::notify_timer(timer) {
                eprintln!("Could not deliver the timer alert: {error:#}");
            }
            return Ok(None);
        }

        if !running.load(Ordering::Relaxed) {
            session.cancel_capture()?;
            return Ok(None);
        }

        match session.recv_level_timeout(Duration::from_secs(1)) {
            Ok(level) => {
                print_level(level)?;

                match detector.observe(level) {
                    Some(vad::VadEvent::Calibrated { noise_floor_dbfs }) => {
                        println!();
                        print_vad_thresholds(detector, noise_floor_dbfs);
                    }
                    Some(vad::VadEvent::SpeechStarted { at_frame }) => {
                        let pre_roll_frames =
                            frames_for_duration(session.sample_rate(), PRE_ROLL_DURATION);
                        speech_start_frame = Some(at_frame.saturating_sub(pre_roll_frames));
                        println!("\nSpeech detected.");
                    }
                    Some(vad::VadEvent::SpeechEnded { at_frame, reason }) => {
                        println!("\nSpeech ended ({reason:?}).");
                        let metrics = detector
                            .metrics()
                            .context("VAD finished without producing telemetry")?;
                        break (at_frame, metrics);
                    }
                    Some(vad::VadEvent::TimedOut) => {
                        if !preserve_missed_capture {
                            session.cancel_capture()?;
                            println!("\nNo speech detected. Continuing to listen...");
                            return Ok(None);
                        }
                        let recording = session
                            .finish_capture()
                            .context("failed to preserve prompted VAD miss")?;
                        let duration_ms = (recording.frame_count() as f64 * 1_000.0
                            / f64::from(recording.sample_rate))
                        .round() as u64;
                        let (start_threshold, end_threshold) = detector
                            .thresholds_dbfs()
                            .context("VAD timed out before calibration")?;
                        println!("\nNo speech detected; preserving prompted false negative.");
                        return Ok(Some(CapturedUtterance {
                            recording,
                            vad: dataset::VadTelemetry {
                                noise_floor_dbfs: detector.noise_floor_dbfs(),
                                start_threshold_dbfs: Some(start_threshold),
                                end_threshold_dbfs: Some(end_threshold),
                                duration_ms: Some(duration_ms),
                                speech_windows: Some(0),
                                end_reason: Some("timeout".to_owned()),
                                ..dataset::VadTelemetry::default()
                            },
                        }));
                    }
                    None => {}
                }
            }
            Err(RecvTimeoutError::Timeout) => anyhow::bail!("audio input stopped producing data"),
            Err(RecvTimeoutError::Disconnected) => anyhow::bail!("audio level stream disconnected"),
        }
    };

    let speech_start_frame = speech_start_frame.context("speech ended before it started")?;
    let recording = session
        .finish_capture()
        .context("failed to finish recording")?;

    let recording = recording.slice_frames(speech_start_frame..speech_end_frame)?;
    let duration_ms = (recording.frame_count() as f64 * 1_000.0 / f64::from(recording.sample_rate))
        .round() as u64;
    let window_ms = LEVEL_WINDOW.as_millis() as u64;

    Ok(Some(CapturedUtterance {
        recording,
        vad: dataset::VadTelemetry {
            noise_floor_dbfs: Some(vad_metrics.noise_floor_dbfs),
            start_threshold_dbfs: Some(vad_metrics.start_threshold_dbfs),
            end_threshold_dbfs: Some(vad_metrics.end_threshold_dbfs),
            peak_dbfs: Some(vad_metrics.peak_dbfs),
            mean_speech_dbfs: Some(vad_metrics.mean_speech_dbfs),
            median_speech_dbfs: Some(vad_metrics.median_speech_dbfs),
            duration_ms: Some(duration_ms),
            trailing_silence_ms: Some(vad_metrics.trailing_silence_windows as u64 * window_ms),
            speech_windows: Some(vad_metrics.speech_windows),
            silence_windows: Some(vad_metrics.silence_windows),
            end_reason: Some(vad_metrics.end_reason.as_str().to_owned()),
        },
    }))
}

fn print_vad_thresholds(detector: &vad::VadDetector, noise_floor: f32) {
    let (start_threshold, end_threshold) = detector
        .thresholds_dbfs()
        .expect("calibrated VAD has thresholds");
    println!(
        "Listening... noise floor: {noise_floor:.1} dBFS, start: {start_threshold:.1} dBFS, end: {end_threshold:.1} dBFS"
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn telemetry_with_speech_windows(speech_windows: usize) -> dataset::VadTelemetry {
        dataset::VadTelemetry {
            speech_windows: Some(speech_windows),
            ..dataset::VadTelemetry::default()
        }
    }

    #[test]
    fn rejects_candidates_shorter_than_one_hundred_forty_milliseconds() {
        assert!(!has_minimum_speech(&telemetry_with_speech_windows(6)));
        assert!(has_minimum_speech(&telemetry_with_speech_windows(7)));
    }

    #[test]
    fn noisy_collection_profile_only_relaxes_adaptive_margins() {
        let normal = vad_config();
        let collection = collection_vad_config();

        assert_eq!(normal.start_margin_db, 12.0);
        assert_eq!(normal.end_margin_db, 6.0);
        assert_eq!(collection.start_margin_db, 6.0);
        assert_eq!(collection.end_margin_db, 2.0);
        assert_eq!(collection.minimum_start_level_dbfs, -35.0);
        assert_eq!(collection.minimum_end_level_dbfs, -40.0);
    }
}
