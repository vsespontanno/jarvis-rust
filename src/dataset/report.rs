use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result};

use super::{
    DatasetLabel, DatasetRecord, LabelStore,
    analysis::{DatasetAnalysis, NumericStatistics, ParserEvaluation, VadStatistics, analyze},
};
use crate::parser;

#[derive(Debug)]
struct DatasetReport {
    events: usize,
    audio_samples: usize,
    transcripts: usize,
    processing_errors: usize,
    short_rejections: usize,
    non_speech_annotations: usize,
    replayed_transcripts: usize,
    changed_predictions: usize,
    resolved_unknowns: usize,
    analysis: DatasetAnalysis,
    execution_statuses: BTreeMap<String, usize>,
    intents: BTreeMap<String, usize>,
    current_intents: BTreeMap<String, usize>,
    prediction_changes: BTreeMap<String, usize>,
    devices: BTreeMap<String, usize>,
    end_reasons: BTreeMap<String, usize>,
    durations_ms: Option<Distribution>,
    speech_windows: Option<Distribution>,
    noise_floor_dbfs: Option<Distribution>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Distribution {
    minimum: f64,
    median: f64,
    percentile_95: f64,
    maximum: f64,
}

pub fn print_report(path: &Path) -> Result<()> {
    let file = File::open(path)
        .with_context(|| format!("failed to open dataset events at {}", path.display()))?;
    let dataset_root = path.parent().unwrap_or_else(|| Path::new("."));
    let labels = LabelStore::open(dataset_root)?.latest()?;
    let report = DatasetReport::from_reader_with_labels(BufReader::new(file), &labels)?;
    println!("Dataset: {}\n{report}", path.display());
    Ok(())
}

impl DatasetReport {
    fn from_reader_with_labels(
        reader: impl BufRead,
        labels: &BTreeMap<String, DatasetLabel>,
    ) -> Result<Self> {
        let mut records = Vec::new();

        for (index, line) in reader.lines().enumerate() {
            let line_number = index + 1;
            let line = line.with_context(|| format!("failed to read JSONL line {line_number}"))?;
            if line.trim().is_empty() {
                continue;
            }

            let record = serde_json::from_str(&line)
                .with_context(|| format!("invalid dataset record on JSONL line {line_number}"))?;
            records.push(record);
        }

        Ok(Self::from_records(&records, labels))
    }

    fn from_records(records: &[DatasetRecord], labels: &BTreeMap<String, DatasetLabel>) -> Self {
        let mut execution_statuses = BTreeMap::new();
        let mut intents = BTreeMap::new();
        let mut current_intents = BTreeMap::new();
        let mut prediction_changes = BTreeMap::new();
        let mut devices = BTreeMap::new();
        let mut end_reasons = BTreeMap::new();
        let mut durations_ms = Vec::new();
        let mut speech_windows = Vec::new();
        let mut noise_floor_dbfs = Vec::new();
        let mut replayed_transcripts = 0;
        let mut changed_predictions = 0;
        let mut resolved_unknowns = 0;

        for record in records {
            increment(
                &mut execution_statuses,
                record.execution_result.status.as_str(),
            );
            increment(&mut devices, record.input.device.trim());

            if let Some(prediction) = &record.prediction {
                increment(&mut intents, &prediction.intent);
            }

            if let Some(transcript) = &record.transcript {
                replayed_transcripts += 1;
                let current_intent = parser::parse(transcript).intent_name();
                increment(&mut current_intents, current_intent);

                if let Some(stored) = &record.prediction
                    && stored.intent != current_intent
                {
                    changed_predictions += 1;
                    if stored.intent == "unknown" && current_intent != "unknown" {
                        resolved_unknowns += 1;
                    }
                    increment(
                        &mut prediction_changes,
                        &format!("{} -> {current_intent}", stored.intent),
                    );
                }
            }

            if let Some(vad) = &record.vad {
                durations_ms.extend(vad.duration_ms.map(|value| value as f64));
                speech_windows.extend(vad.speech_windows.map(|value| value as f64));
                noise_floor_dbfs.extend(vad.noise_floor_dbfs.map(f64::from));
                if let Some(reason) = &vad.end_reason {
                    increment(&mut end_reasons, reason);
                }
            }
        }

        Self {
            events: records.len(),
            audio_samples: records
                .iter()
                .filter(|record| record.audio_path.is_some())
                .count(),
            transcripts: records
                .iter()
                .filter(|record| record.transcript.is_some())
                .count(),
            processing_errors: records
                .iter()
                .filter(|record| record.processing_error.is_some())
                .count(),
            short_rejections: records
                .iter()
                .filter(|record| {
                    record.execution_result.status == super::ExecutionStatus::Rejected
                        && record.transcript.is_none()
                })
                .count(),
            non_speech_annotations: records
                .iter()
                .filter(|record| {
                    record
                        .prediction
                        .as_ref()
                        .is_some_and(|prediction| prediction.intent == "no_speech")
                })
                .count(),
            replayed_transcripts,
            changed_predictions,
            resolved_unknowns,
            analysis: analyze(records, labels),
            execution_statuses,
            intents,
            current_intents,
            prediction_changes,
            devices,
            end_reasons,
            durations_ms: Distribution::from_values(durations_ms),
            speech_windows: Distribution::from_values(speech_windows),
            noise_floor_dbfs: Distribution::from_values(noise_floor_dbfs),
        }
    }
}

impl Distribution {
    fn from_values(mut values: Vec<f64>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        values.sort_unstable_by(f64::total_cmp);
        let middle = values.len() / 2;
        let median = if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        };
        let percentile_95_index = (values.len() * 95).div_ceil(100).saturating_sub(1);

        Some(Self {
            minimum: values[0],
            median,
            percentile_95: values[percentile_95_index],
            maximum: values[values.len() - 1],
        })
    }
}

impl fmt::Display for DatasetReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "Events: {}", self.events)?;
        writeln!(formatter, "Audio samples: {}", self.audio_samples)?;
        writeln!(formatter, "Transcripts: {}", self.transcripts)?;
        writeln!(formatter, "Processing errors: {}", self.processing_errors)?;
        writeln!(formatter, "Short rejections: {}", self.short_rejections)?;
        writeln!(
            formatter,
            "Whisper non-speech annotations: {}",
            self.non_speech_annotations
        )?;
        writeln!(
            formatter,
            "Replayed transcripts: {}",
            self.replayed_transcripts
        )?;
        writeln!(
            formatter,
            "Changed predictions: {}",
            self.changed_predictions
        )?;
        writeln!(
            formatter,
            "Resolved historical unknowns: {}",
            self.resolved_unknowns
        )?;
        write_pipeline_analysis(formatter, &self.analysis, self.events)?;
        write_counts(formatter, "Execution", &self.execution_statuses)?;
        write_counts(formatter, "Stored intents", &self.intents)?;
        write_counts(formatter, "Current parser intents", &self.current_intents)?;
        write_counts(formatter, "Prediction changes", &self.prediction_changes)?;
        write_counts(formatter, "Input devices", &self.devices)?;
        write_counts(formatter, "VAD end reasons", &self.end_reasons)?;

        writeln!(formatter, "\nVAD distributions (min / median / p95 / max):")?;
        write_distribution(formatter, "duration", self.durations_ms, "ms")?;
        write_distribution(formatter, "speech windows", self.speech_windows, "")?;
        write_distribution(formatter, "noise floor", self.noise_floor_dbfs, "dBFS")
    }
}

fn write_pipeline_analysis(
    formatter: &mut fmt::Formatter<'_>,
    analysis: &DatasetAnalysis,
    events: usize,
) -> fmt::Result {
    let detection = &analysis.detection;
    writeln!(
        formatter,
        "\nLabeled pipeline evaluation: {} / {events}",
        analysis.labeled
    )?;
    writeln!(formatter, "\nSpeech detection:")?;
    writeln!(formatter, "  Actual speech: {}", detection.actual_speech)?;
    writeln!(
        formatter,
        "  Actual non-speech / false activations: {}",
        detection.actual_non_speech
    )?;
    writeln!(
        formatter,
        "  Non-speech rejected before Whisper: {}",
        detection.non_speech_rejected_before_stt
    )?;
    writeln!(
        formatter,
        "  Non-speech that reached Whisper: {}",
        detection.non_speech_reached_stt
    )?;
    writeln!(
        formatter,
        "  Non-speech with non-empty Whisper text: {}",
        detection.non_speech_with_text
    )?;
    writeln!(
        formatter,
        "  Non-speech recognized as sound annotations: {}",
        detection.non_speech_annotations
    )?;

    let stt = &analysis.stt;
    writeln!(formatter, "\nWhisper (actual speech only):")?;
    writeln!(formatter, "  Speech samples: {}", stt.speech_samples)?;
    writeln!(formatter, "  Samples evaluated: {}", stt.evaluated_samples)?;
    writeln!(
        formatter,
        "  Samples with word errors: {}",
        stt.samples_with_errors
    )?;
    writeln!(
        formatter,
        "  Word errors (substitutions / deletions / insertions): {} / {} / {}",
        stt.substitutions, stt.deletions, stt.insertions
    )?;
    write_rate(formatter, "Aggregate WER", stt.aggregate_wer)?;
    write_rate(formatter, "Mean per-sample WER", stt.mean_sample_wer)?;
    write_rate(formatter, "Median per-sample WER", stt.median_sample_wer)?;

    write_parser_evaluation(formatter, "Historical parser", &analysis.historical_parser)?;
    write_parser_evaluation(formatter, "Current parser", &analysis.current_parser)?;

    write_vad_statistics(
        formatter,
        "VAD telemetry — actual speech",
        &analysis.speech_vad,
    )?;
    write_vad_statistics(
        formatter,
        "VAD telemetry — actual non-speech",
        &analysis.non_speech_vad,
    )?;

    writeln!(formatter, "\nMinimum speech duration simulation:")?;
    writeln!(
        formatter,
        "  mark threshold  speech kept/rejected  noise kept/rejected  recall  noise rejection"
    )?;
    for row in &analysis.thresholds {
        let highlighted = matches!(row.threshold_ms, 100 | 140 | 200 | 300);
        let candidate = analysis.candidate_thresholds.contains(&row.threshold_ms);
        let mark = match (highlighted, candidate) {
            (true, true) => "*!",
            (true, false) => "* ",
            (false, true) => " !",
            (false, false) => "  ",
        };
        writeln!(
            formatter,
            "  {mark} {:>3} ms      {:>3}/{:<3}             {:>3}/{:<3}          {:>6}  {:>6}",
            row.threshold_ms,
            row.speech_kept,
            row.speech_rejected,
            row.non_speech_kept,
            row.non_speech_rejected,
            format_rate(row.speech_recall),
            format_rate(row.noise_rejection_rate),
        )?;
    }
    writeln!(
        formatter,
        "  * requested checkpoint; ! data-based candidate"
    )?;
    if analysis.candidate_thresholds.is_empty() {
        writeln!(formatter, "  Candidates: no labeled VAD data")
    } else {
        let candidates = analysis
            .candidate_thresholds
            .iter()
            .map(|threshold| format!("{threshold} ms"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            formatter,
            "  Candidates: {candidates} (configuration unchanged)"
        )
    }
}

fn write_parser_evaluation(
    formatter: &mut fmt::Formatter<'_>,
    heading: &str,
    evaluation: &ParserEvaluation,
) -> fmt::Result {
    writeln!(formatter, "\n{heading} (supported labeled commands only):")?;
    writeln!(
        formatter,
        "  Correct: {} / {} ({})",
        evaluation.correct,
        evaluation.total,
        format_rate(evaluation.accuracy())
    )?;
    writeln!(formatter, "  Confusion (expected -> predicted):")?;
    if evaluation.confusion.is_empty() {
        writeln!(formatter, "    (no data)")?;
    } else {
        for (expected, predictions) in &evaluation.confusion {
            for (predicted, count) in predictions {
                writeln!(formatter, "    {expected} -> {predicted}: {count}")?;
            }
        }
    }
    writeln!(formatter, "  Errors:")?;
    if evaluation.mistakes.is_empty() {
        writeln!(formatter, "    (none)")?;
    } else {
        for mistake in &evaluation.mistakes {
            writeln!(
                formatter,
                "    {}: {} -> {} | \"{}\"",
                mistake.event_id, mistake.expected, mistake.predicted, mistake.transcript
            )?;
        }
    }
    Ok(())
}

fn write_vad_statistics(
    formatter: &mut fmt::Formatter<'_>,
    heading: &str,
    statistics: &VadStatistics,
) -> fmt::Result {
    writeln!(
        formatter,
        "\n{heading} (count / min / median / mean / max):"
    )?;
    write_numeric_statistics(formatter, "duration", statistics.duration_ms, "ms")?;
    write_numeric_statistics(formatter, "speech windows", statistics.speech_windows, "")?;
    write_numeric_statistics(formatter, "peak", statistics.peak_dbfs, "dBFS")?;
    write_numeric_statistics(
        formatter,
        "mean speech level",
        statistics.mean_speech_dbfs,
        "dBFS",
    )?;
    write_numeric_statistics(
        formatter,
        "median speech level",
        statistics.median_speech_dbfs,
        "dBFS",
    )?;
    write_numeric_statistics(
        formatter,
        "noise floor",
        statistics.noise_floor_dbfs,
        "dBFS",
    )?;
    write_numeric_statistics(
        formatter,
        "start threshold",
        statistics.start_threshold_dbfs,
        "dBFS",
    )?;
    write_numeric_statistics(
        formatter,
        "end threshold",
        statistics.end_threshold_dbfs,
        "dBFS",
    )
}

fn write_numeric_statistics(
    formatter: &mut fmt::Formatter<'_>,
    name: &str,
    statistics: Option<NumericStatistics>,
    unit: &str,
) -> fmt::Result {
    let Some(statistics) = statistics else {
        return writeln!(formatter, "  {name}: no data");
    };
    let separator = if unit.is_empty() { "" } else { " " };
    writeln!(
        formatter,
        "  {name}: {} / {:.1}{separator}{unit} / {:.1}{separator}{unit} / {:.1}{separator}{unit} / {:.1}{separator}{unit}",
        statistics.count,
        statistics.minimum,
        statistics.median,
        statistics.mean,
        statistics.maximum,
    )
}

fn write_rate(formatter: &mut fmt::Formatter<'_>, name: &str, rate: Option<f64>) -> fmt::Result {
    writeln!(formatter, "  {name}: {}", format_rate(rate))
}

fn format_rate(rate: Option<f64>) -> String {
    rate.map_or_else(|| "n/a".to_owned(), |rate| format!("{:.1}%", rate * 100.0))
}

fn increment(counts: &mut BTreeMap<String, usize>, key: &str) {
    *counts.entry(key.to_owned()).or_default() += 1;
}

fn write_counts(
    formatter: &mut fmt::Formatter<'_>,
    heading: &str,
    counts: &BTreeMap<String, usize>,
) -> fmt::Result {
    writeln!(formatter, "\n{heading}:")?;
    if counts.is_empty() {
        return writeln!(formatter, "  (no data)");
    }
    for (name, count) in counts {
        writeln!(formatter, "  {name}: {count}")?;
    }
    Ok(())
}

fn write_distribution(
    formatter: &mut fmt::Formatter<'_>,
    name: &str,
    distribution: Option<Distribution>,
    unit: &str,
) -> fmt::Result {
    let Some(distribution) = distribution else {
        return writeln!(formatter, "  {name}: no data");
    };
    let separator = if unit.is_empty() { "" } else { " " };

    writeln!(
        formatter,
        "  {name}: {:.1}{separator}{unit} / {:.1}{separator}{unit} / {:.1}{separator}{unit} / {:.1}{separator}{unit}",
        distribution.minimum, distribution.median, distribution.percentile_95, distribution.maximum,
    )
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, io::Cursor};

    use super::*;
    use crate::dataset::{
        ExecutionResult, ExecutionStatus, GroundTruth, InputMetadata, IntentPrediction,
        VadTelemetry,
    };

    fn record(
        id: &str,
        status: ExecutionStatus,
        intent: Option<&str>,
        speech_windows: usize,
    ) -> DatasetRecord {
        DatasetRecord {
            schema_version: 1,
            id: id.to_owned(),
            timestamp: "2026-09-10T00:00:00Z".to_owned(),
            provenance: None,
            audio_path: (speech_windows >= 7).then(|| format!("utterances/{id}.wav")),
            input: InputMetadata {
                device: "Test microphone".to_owned(),
                sample_rate_hz: 24_000,
                channels: 1,
            },
            transcript: intent.map(|_| "example".to_owned()),
            prediction: intent.map(|intent| IntentPrediction {
                intent: intent.to_owned(),
                slots: BTreeMap::new(),
            }),
            execution_result: ExecutionResult {
                status,
                response: None,
                error: None,
            },
            processing_error: None,
            vad: Some(VadTelemetry {
                noise_floor_dbfs: Some(-50.0),
                duration_ms: Some(speech_windows as u64 * 20),
                speech_windows: Some(speech_windows),
                end_reason: Some("silence".to_owned()),
                ..VadTelemetry::default()
            }),
            ground_truth: GroundTruth::default(),
        }
    }

    #[test]
    fn aggregates_json_lines_records() {
        let mut records = [
            record("short", ExecutionStatus::Rejected, None, 4),
            record("time", ExecutionStatus::Succeeded, Some("tell_time"), 23),
            record("noise", ExecutionStatus::Rejected, Some("no_speech"), 38),
        ];
        records[0].input.device.push(' ');
        let jsonl = records
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join("\n");

        let report =
            DatasetReport::from_reader_with_labels(Cursor::new(jsonl), &BTreeMap::new()).unwrap();

        assert_eq!(report.events, 3);
        assert_eq!(report.audio_samples, 2);
        assert_eq!(report.short_rejections, 1);
        assert_eq!(report.non_speech_annotations, 1);
        assert_eq!(report.execution_statuses["rejected"], 2);
        assert_eq!(report.intents["tell_time"], 1);
        assert_eq!(report.devices["Test microphone"], 3);
        assert_eq!(report.speech_windows.unwrap().median, 23.0);
    }

    #[test]
    fn calculates_median_and_nearest_rank_percentile() {
        assert_eq!(
            Distribution::from_values(vec![1.0, 2.0, 3.0, 4.0]),
            Some(Distribution {
                minimum: 1.0,
                median: 2.5,
                percentile_95: 4.0,
                maximum: 4.0,
            })
        );
    }

    #[test]
    fn replays_historical_prediction_with_current_parser() {
        let mut historical = record(
            "historical",
            ExecutionStatus::Unsupported,
            Some("unknown"),
            20,
        );
        historical.transcript = Some("Открою Spotify.".to_owned());

        let report = DatasetReport::from_records(&[historical], &BTreeMap::new());

        assert_eq!(report.replayed_transcripts, 1);
        assert_eq!(report.changed_predictions, 1);
        assert_eq!(report.resolved_unknowns, 1);
        assert_eq!(report.current_intents["play_music"], 1);
        assert_eq!(report.prediction_changes["unknown -> play_music"], 1);
    }

    #[test]
    fn joins_human_labels_without_changing_historical_predictions() {
        let time = record("time", ExecutionStatus::Succeeded, Some("tell_time"), 23);
        let mut noise = record("noise", ExecutionStatus::Rejected, Some("unknown"), 10);
        noise.transcript = Some("[музыка]".to_owned());
        let labels = BTreeMap::from([
            (
                "time".to_owned(),
                DatasetLabel::new(
                    "time".to_owned(),
                    true,
                    Some("example".to_owned()),
                    "tell_time".to_owned(),
                    None,
                ),
            ),
            (
                "noise".to_owned(),
                DatasetLabel::new(
                    "noise".to_owned(),
                    false,
                    None,
                    "no_speech".to_owned(),
                    Some("table knock".to_owned()),
                ),
            ),
        ]);

        let report = DatasetReport::from_records(&[time, noise], &labels);

        assert_eq!(report.analysis.labeled, 2);
        assert_eq!(report.analysis.detection.actual_speech, 1);
        assert_eq!(report.analysis.detection.actual_non_speech, 1);
        assert_eq!(report.analysis.detection.non_speech_reached_stt, 1);
        assert_eq!(report.analysis.historical_parser.correct, 1);
        assert_eq!(report.intents["unknown"], 1);
    }

    #[test]
    fn reports_invalid_line_number() {
        let error =
            DatasetReport::from_reader_with_labels(Cursor::new("{}\nnot-json\n"), &BTreeMap::new())
                .unwrap_err();

        assert!(error.to_string().contains("JSONL line 1"));
    }
}
