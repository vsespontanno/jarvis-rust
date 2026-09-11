use std::{
    cmp::Reverse,
    collections::BTreeMap,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use super::{
    DatasetLabel, DatasetRecord, LabelStore,
    analysis::{WordErrors, normalize_transcript, word_errors},
};
use crate::{audio::Recording, parser, preprocessing, stt::WhisperTranscriber};

const BENCHMARK_SCHEMA_VERSION: u8 = 1;
const WORST_CASES_LIMIT: usize = 10;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct WordErrorMetrics {
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    reference_words: usize,
    wer: Option<f64>,
}

impl From<WordErrors> for WordErrorMetrics {
    fn from(errors: WordErrors) -> Self {
        Self {
            substitutions: errors.substitutions,
            deletions: errors.deletions,
            insertions: errors.insertions,
            reference_words: errors.reference_words,
            wer: errors.rate(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct BenchmarkRecord {
    schema_version: u8,
    event_id: String,
    session_id: Option<String>,
    campaign: Option<String>,
    device: String,
    audio_path: String,
    model_path: String,
    reference_transcript: String,
    expected_intent: String,
    historical_transcript: Option<String>,
    historical_intent: Option<String>,
    benchmark_transcript: Option<String>,
    benchmark_intent: Option<String>,
    latency_ms: Option<u64>,
    word_errors: Option<WordErrorMetrics>,
    spotify_case: bool,
    vad_end_reason: Option<String>,
    duration_ms: Option<u64>,
    error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct NumericSummary {
    minimum: f64,
    median: f64,
    mean: f64,
    maximum: f64,
}

#[derive(Debug, Default, PartialEq)]
struct BenchmarkSummary {
    total: usize,
    completed: usize,
    failed: usize,
    exact: usize,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    reference_words: usize,
    aggregate_wer: Option<f64>,
    mean_sample_wer: Option<f64>,
    median_sample_wer: Option<f64>,
    latency_ms: Option<NumericSummary>,
    spotify_total: usize,
    spotify_completed: usize,
    spotify_recognized: usize,
    spotify_aggregate_wer: Option<f64>,
    intent_correct: usize,
    supported_intents: usize,
    supported_intents_correct: usize,
    negative_intents: usize,
    false_commands: usize,
}

pub fn benchmark_stt(root: &Path, model_path: &Path, campaign: Option<&str>) -> Result<()> {
    let events = read_events(&root.join("events.jsonl"))?;
    let labels = LabelStore::open(root)?.latest()?;
    let samples = select_samples(root, &events, &labels, campaign);
    ensure!(
        !samples.is_empty(),
        "no labeled speech samples with audio matched the benchmark filter"
    );

    let output_path = benchmark_output_path(root, model_path, campaign);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    println!("Loading benchmark model from {}...", model_path.display());
    let transcriber = WhisperTranscriber::load(model_path)
        .context("failed to load the benchmark speech recognition model")?;
    println!(
        "Benchmarking {} sample(s){}...",
        samples.len(),
        campaign.map_or(String::new(), |value| format!(" from campaign '{value}'"))
    );

    let mut results = Vec::with_capacity(samples.len());
    for (index, (event, label)) in samples.iter().enumerate() {
        println!("[{}/{}] {}", index + 1, samples.len(), event.id);
        results.push(benchmark_sample(
            root,
            model_path,
            event,
            label,
            &transcriber,
        ));
    }

    write_results(&output_path, &results)?;
    let summary = summarize(&results);
    print_summary(&output_path, &results, &summary);
    Ok(())
}

fn read_events(path: &Path) -> Result<Vec<DatasetRecord>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut events = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.with_context(|| format!("failed to read JSONL line {line_number}"))?;
        if line.trim().is_empty() {
            continue;
        }
        events.push(
            serde_json::from_str(&line)
                .with_context(|| format!("invalid dataset event on JSONL line {line_number}"))?,
        );
    }
    Ok(events)
}

fn select_samples<'a>(
    root: &Path,
    events: &'a [DatasetRecord],
    labels: &'a BTreeMap<String, DatasetLabel>,
    campaign: Option<&str>,
) -> Vec<(&'a DatasetRecord, &'a DatasetLabel)> {
    events
        .iter()
        .filter(|event| {
            campaign.is_none_or(|expected| {
                event
                    .collection
                    .as_ref()
                    .is_some_and(|collection| collection.campaign == expected)
            })
        })
        .filter_map(|event| labels.get(&event.id).map(|label| (event, label)))
        .filter(|(event, label)| {
            label.actual_speech
                && label
                    .corrected_transcript
                    .as_deref()
                    .is_some_and(|text| !text.trim().is_empty())
                && event
                    .audio_path
                    .as_deref()
                    .is_some_and(|path| root.join(path).is_file())
        })
        .collect()
}

fn benchmark_sample(
    root: &Path,
    model_path: &Path,
    event: &DatasetRecord,
    label: &DatasetLabel,
    transcriber: &WhisperTranscriber,
) -> BenchmarkRecord {
    let audio_path = event
        .audio_path
        .as_deref()
        .expect("selected benchmark sample has audio");
    let reference = label
        .corrected_transcript
        .as_deref()
        .expect("selected benchmark sample has a reference");
    let mut result = BenchmarkRecord {
        schema_version: BENCHMARK_SCHEMA_VERSION,
        event_id: event.id.clone(),
        session_id: event
            .provenance
            .as_ref()
            .map(|provenance| provenance.session_id.clone()),
        campaign: event
            .collection
            .as_ref()
            .map(|collection| collection.campaign.clone()),
        device: event.input.device.clone(),
        audio_path: audio_path.to_owned(),
        model_path: model_path.display().to_string(),
        reference_transcript: reference.to_owned(),
        expected_intent: label.correct_intent.clone(),
        historical_transcript: event.transcript.clone(),
        historical_intent: event
            .prediction
            .as_ref()
            .map(|prediction| prediction.intent.clone()),
        benchmark_transcript: None,
        benchmark_intent: None,
        latency_ms: None,
        word_errors: None,
        spotify_case: is_spotify_case(reference),
        vad_end_reason: event.vad.as_ref().and_then(|vad| vad.end_reason.clone()),
        duration_ms: event.vad.as_ref().and_then(|vad| vad.duration_ms),
        error: None,
    };

    let benchmark = (|| -> Result<(String, u64)> {
        let recording = read_wav(&root.join(audio_path), &event.input.device)?;
        let prepared = preprocessing::prepare_for_stt(&recording)
            .context("failed to prepare benchmark audio")?;
        let started = Instant::now();
        let transcript = transcriber
            .transcribe(&prepared.samples)
            .context("benchmark transcription failed")?;
        let latency_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        Ok((transcript, latency_ms))
    })();

    match benchmark {
        Ok((transcript, latency_ms)) => {
            result.word_errors = Some(word_errors(reference, &transcript).into());
            result.benchmark_intent = Some(parser::parse(&transcript).intent_name().to_owned());
            result.benchmark_transcript = Some(transcript);
            result.latency_ms = Some(latency_ms);
        }
        Err(error) => result.error = Some(format!("{error:#}")),
    }
    result
}

fn read_wav(path: &Path, device: &str) -> Result<Recording> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open benchmark WAV {}", path.display()))?;
    let spec = reader.spec();
    ensure!(spec.channels > 0, "WAV has no channels");
    ensure!(spec.sample_rate > 0, "WAV has no sample rate");
    ensure!(
        spec.sample_format == hound::SampleFormat::Int && spec.bits_per_sample == 16,
        "unsupported WAV format; expected signed 16-bit PCM"
    );
    let samples = reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to decode benchmark WAV")?;
    ensure!(
        samples.len().is_multiple_of(spec.channels as usize),
        "WAV ends with an incomplete frame"
    );
    Ok(Recording {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        device_name: device.to_owned(),
    })
}

fn write_results(path: &Path, results: &[BenchmarkRecord]) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create benchmark output {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for result in results {
        serde_json::to_writer(&mut writer, result)
            .context("failed to serialize benchmark result")?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn benchmark_output_path(root: &Path, model_path: &Path, campaign: Option<&str>) -> PathBuf {
    let model = model_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    let scope = campaign.unwrap_or("all");
    root.join("benchmarks").join(format!(
        "{}-{}.jsonl",
        safe_filename(model),
        safe_filename(scope)
    ))
}

fn safe_filename(value: &str) -> String {
    let value = value
        .chars()
        .map(|character| {
            if character.is_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    value.trim_matches('-').to_owned()
}

fn is_spotify_case(reference: &str) -> bool {
    let normalized = normalize_transcript(reference);
    normalized.contains("spotify") || normalized.contains("спотиф") || normalized.contains("спотик")
}

fn summarize(results: &[BenchmarkRecord]) -> BenchmarkSummary {
    let mut summary = BenchmarkSummary {
        total: results.len(),
        ..BenchmarkSummary::default()
    };
    let mut rates = Vec::new();
    let mut latencies = Vec::new();
    let mut spotify_edits = 0;
    let mut spotify_words = 0;

    for result in results {
        if result.spotify_case {
            summary.spotify_total += 1;
        }
        let Some(errors) = &result.word_errors else {
            summary.failed += 1;
            continue;
        };
        summary.completed += 1;
        summary.substitutions += errors.substitutions;
        summary.deletions += errors.deletions;
        summary.insertions += errors.insertions;
        summary.reference_words += errors.reference_words;
        summary.exact +=
            usize::from(errors.substitutions + errors.deletions + errors.insertions == 0);
        rates.extend(errors.wer);
        latencies.extend(result.latency_ms.map(|value| value as f64));
        if result.spotify_case {
            summary.spotify_completed += 1;
            summary.spotify_recognized += usize::from(
                result
                    .benchmark_transcript
                    .as_deref()
                    .is_some_and(is_spotify_case),
            );
            spotify_edits += errors.substitutions + errors.deletions + errors.insertions;
            spotify_words += errors.reference_words;
        }
        let predicted = result.benchmark_intent.as_deref().unwrap_or("unknown");
        summary.intent_correct += usize::from(predicted == result.expected_intent);
        if is_actionable_intent(&result.expected_intent) {
            summary.supported_intents += 1;
            summary.supported_intents_correct += usize::from(predicted == result.expected_intent);
        } else {
            summary.negative_intents += 1;
            summary.false_commands += usize::from(is_actionable_intent(predicted));
        }
    }

    let total_edits = summary.substitutions + summary.deletions + summary.insertions;
    summary.aggregate_wer =
        (summary.reference_words > 0).then(|| total_edits as f64 / summary.reference_words as f64);
    summary.mean_sample_wer = mean(&rates);
    summary.median_sample_wer = median(&mut rates);
    summary.latency_ms = numeric_summary(&mut latencies);
    summary.spotify_aggregate_wer =
        (spotify_words > 0).then(|| spotify_edits as f64 / spotify_words as f64);
    summary
}

fn numeric_summary(values: &mut [f64]) -> Option<NumericSummary> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    Some(NumericSummary {
        minimum: values[0],
        median: median_sorted(values),
        mean: values.iter().sum::<f64>() / values.len() as f64,
        maximum: values[values.len() - 1],
    })
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    Some(median_sorted(values))
}

fn median_sorted(values: &[f64]) -> f64 {
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn print_summary(path: &Path, results: &[BenchmarkRecord], summary: &BenchmarkSummary) {
    println!("\nSTT benchmark: {}", path.display());
    println!("Samples: {}", summary.total);
    println!(
        "Completed / failed: {} / {}",
        summary.completed, summary.failed
    );
    println!("Exact transcripts: {}", summary.exact);
    println!(
        "Word errors (substitutions / deletions / insertions): {} / {} / {}",
        summary.substitutions, summary.deletions, summary.insertions
    );
    println!("Aggregate WER: {}", format_rate(summary.aggregate_wer));
    println!(
        "Historical aggregate WER on the same samples: {}",
        format_rate(historical_wer(results))
    );
    let (historical_intent_correct, historical_intent_total) = historical_intent_accuracy(results);
    println!(
        "Historical intent accuracy on the same samples: {} / {} ({})",
        historical_intent_correct,
        historical_intent_total,
        format_fraction(historical_intent_correct, historical_intent_total)
    );
    println!("Mean sample WER: {}", format_rate(summary.mean_sample_wer));
    println!(
        "Median sample WER: {}",
        format_rate(summary.median_sample_wer)
    );
    if let Some(latency) = summary.latency_ms {
        println!(
            "Latency (min / median / mean / max): {:.0} / {:.0} / {:.0} / {:.0} ms",
            latency.minimum, latency.median, latency.mean, latency.maximum
        );
    }
    println!(
        "Spotify cases: {} / {} completed, brand retained in {} / {}, aggregate WER {}",
        summary.spotify_completed,
        summary.spotify_total,
        summary.spotify_recognized,
        summary.spotify_completed,
        format_rate(summary.spotify_aggregate_wer)
    );
    println!(
        "Intent accuracy: {} / {} ({})",
        summary.intent_correct,
        summary.completed,
        format_fraction(summary.intent_correct, summary.completed)
    );
    println!(
        "Supported-command accuracy: {} / {} ({})",
        summary.supported_intents_correct,
        summary.supported_intents,
        format_fraction(summary.supported_intents_correct, summary.supported_intents)
    );
    println!(
        "False commands on negatives: {} / {} ({})",
        summary.false_commands,
        summary.negative_intents,
        format_fraction(summary.false_commands, summary.negative_intents)
    );

    print_grouped_summaries("Sessions", results, |result| {
        result.session_id.as_deref().unwrap_or("<legacy>")
    });
    print_grouped_summaries("Devices", results, |result| &result.device);
    print_grouped_summaries("Campaigns", results, |result| {
        result.campaign.as_deref().unwrap_or("<natural>")
    });
    print_grouped_summaries("VAD end reasons", results, |result| {
        result.vad_end_reason.as_deref().unwrap_or("<missing>")
    });

    let mut worst = results
        .iter()
        .filter_map(|result| {
            result
                .word_errors
                .as_ref()
                .and_then(|errors| errors.wer.map(|rate| (result, rate)))
        })
        .collect::<Vec<_>>();
    worst.sort_by_key(|(_, rate)| Reverse((rate * 1_000_000.0) as u64));
    println!("\nWorst cases:");
    for (result, rate) in worst.into_iter().take(WORST_CASES_LIMIT) {
        println!(
            "  {} | {:.1}% | \"{}\" -> \"{}\"",
            result.event_id,
            rate * 100.0,
            result.reference_transcript,
            result.benchmark_transcript.as_deref().unwrap_or("")
        );
    }

    println!("\nSpotify cases:");
    for result in results.iter().filter(|result| result.spotify_case) {
        println!(
            "  {} | \"{}\" -> \"{}\"",
            result.event_id,
            result.reference_transcript,
            result.benchmark_transcript.as_deref().unwrap_or("<failed>")
        );
    }
}

fn is_actionable_intent(intent: &str) -> bool {
    !matches!(intent, "unknown" | "no_speech")
}

fn historical_wer(results: &[BenchmarkRecord]) -> Option<f64> {
    let (edits, words) = results
        .iter()
        .filter_map(|result| {
            result.historical_transcript.as_deref().map(|transcript| {
                let errors = word_errors(&result.reference_transcript, transcript);
                (errors.edits(), errors.reference_words)
            })
        })
        .fold((0, 0), |(edits, words), (next_edits, next_words)| {
            (edits + next_edits, words + next_words)
        });
    (words > 0).then(|| edits as f64 / words as f64)
}

fn historical_intent_accuracy(results: &[BenchmarkRecord]) -> (usize, usize) {
    results
        .iter()
        .filter_map(|result| {
            result
                .historical_intent
                .as_deref()
                .map(|intent| intent == result.expected_intent)
        })
        .fold((0, 0), |(correct, total), matches| {
            (correct + usize::from(matches), total + 1)
        })
}

fn print_grouped_summaries<'a>(
    title: &str,
    results: &'a [BenchmarkRecord],
    key: impl Fn(&'a BenchmarkRecord) -> &'a str,
) {
    let mut groups: BTreeMap<&str, Vec<BenchmarkRecord>> = BTreeMap::new();
    for result in results {
        groups.entry(key(result)).or_default().push(result.clone());
    }
    println!("\n{title}:");
    for (name, group) in groups {
        let summary = summarize(&group);
        let median_latency = summary.latency_ms.map_or_else(
            || "n/a".to_owned(),
            |latency| format!("{:.0} ms", latency.median),
        );
        println!(
            "  {name}: {} samples, WER {}, median latency {median_latency}",
            summary.completed,
            format_rate(summary.aggregate_wer)
        );
    }
}

fn format_rate(rate: Option<f64>) -> String {
    rate.map_or_else(
        || "n/a".to_owned(),
        |value| format!("{:.1}%", value * 100.0),
    )
}

fn format_fraction(numerator: usize, denominator: usize) -> String {
    format_rate((denominator > 0).then(|| numerator as f64 / denominator as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(reference: &str, hypothesis: &str, latency_ms: u64) -> BenchmarkRecord {
        BenchmarkRecord {
            schema_version: 1,
            event_id: reference.to_owned(),
            session_id: None,
            campaign: None,
            device: "test".to_owned(),
            audio_path: "utterances/test.wav".to_owned(),
            model_path: "model.bin".to_owned(),
            reference_transcript: reference.to_owned(),
            expected_intent: parser::parse(reference).intent_name().to_owned(),
            historical_transcript: None,
            historical_intent: None,
            benchmark_transcript: Some(hypothesis.to_owned()),
            benchmark_intent: Some(parser::parse(hypothesis).intent_name().to_owned()),
            latency_ms: Some(latency_ms),
            word_errors: Some(word_errors(reference, hypothesis).into()),
            spotify_case: is_spotify_case(reference),
            vad_end_reason: Some("silence".to_owned()),
            duration_ms: Some(1_000),
            error: None,
        }
    }

    #[test]
    fn recognizes_latin_and_cyrillic_spotify_cases() {
        assert!(is_spotify_case("открой Spotify"));
        assert!(is_spotify_case("запусти спотифай"));
        assert!(is_spotify_case("включи спотик"));
        assert!(!is_spotify_case("включи музыку"));
    }

    #[test]
    fn summarizes_word_errors_and_latency() {
        let results = [
            result("который час", "который час", 100),
            result("открой Spotify", "открой спотик", 300),
        ];

        let summary = summarize(&results);

        assert_eq!(summary.total, 2);
        assert_eq!(summary.completed, 2);
        assert_eq!(summary.exact, 1);
        assert_eq!(summary.substitutions, 1);
        assert_eq!(summary.reference_words, 4);
        assert_eq!(summary.aggregate_wer, Some(0.25));
        assert_eq!(summary.mean_sample_wer, Some(0.25));
        assert_eq!(summary.median_sample_wer, Some(0.25));
        assert_eq!(summary.latency_ms.unwrap().median, 200.0);
        assert_eq!(summary.spotify_total, 1);
        assert_eq!(summary.spotify_recognized, 1);
        assert_eq!(summary.spotify_aggregate_wer, Some(0.5));
        assert_eq!(historical_wer(&results), None);
    }

    #[test]
    fn builds_stable_safe_output_path() {
        assert_eq!(
            benchmark_output_path(
                Path::new("data"),
                Path::new("models/ggml-small.bin"),
                Some("intent ru/v1")
            ),
            PathBuf::from("data/benchmarks/ggml-small-intent-ru-v1.jsonl")
        );
    }
}
