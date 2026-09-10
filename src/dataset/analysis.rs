use std::collections::BTreeMap;

use super::{DatasetLabel, DatasetRecord, ExecutionStatus};
use crate::parser;

const LEGACY_LEVEL_WINDOW_MS: u64 = 20;

#[derive(Debug)]
pub(super) struct DatasetAnalysis {
    pub labeled: usize,
    pub detection: DetectionEvaluation,
    pub stt: SttEvaluation,
    pub historical_parser: ParserEvaluation,
    pub current_parser: ParserEvaluation,
    pub speech_vad: VadStatistics,
    pub non_speech_vad: VadStatistics,
    pub thresholds: Vec<ThresholdEvaluation>,
    pub candidate_thresholds: Vec<u64>,
}

#[derive(Debug, Default)]
pub(super) struct DetectionEvaluation {
    pub actual_speech: usize,
    pub actual_non_speech: usize,
    pub non_speech_reached_stt: usize,
    pub non_speech_rejected_before_stt: usize,
    pub non_speech_with_text: usize,
    pub non_speech_annotations: usize,
}

#[derive(Debug, Default)]
pub(super) struct SttEvaluation {
    pub speech_samples: usize,
    pub evaluated_samples: usize,
    pub samples_with_errors: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    pub reference_words: usize,
    pub aggregate_wer: Option<f64>,
    pub mean_sample_wer: Option<f64>,
    pub median_sample_wer: Option<f64>,
}

#[derive(Debug, Default)]
pub(super) struct ParserEvaluation {
    pub correct: usize,
    pub total: usize,
    pub confusion: BTreeMap<String, BTreeMap<String, usize>>,
    pub mistakes: Vec<ParserMistake>,
}

#[derive(Debug)]
pub(super) struct ParserMistake {
    pub event_id: String,
    pub expected: String,
    pub predicted: String,
    pub transcript: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct NumericStatistics {
    pub count: usize,
    pub minimum: f64,
    pub median: f64,
    pub mean: f64,
    pub maximum: f64,
}

#[derive(Debug, Default)]
pub(super) struct VadStatistics {
    pub duration_ms: Option<NumericStatistics>,
    pub speech_windows: Option<NumericStatistics>,
    pub peak_dbfs: Option<NumericStatistics>,
    pub mean_speech_dbfs: Option<NumericStatistics>,
    pub median_speech_dbfs: Option<NumericStatistics>,
    pub noise_floor_dbfs: Option<NumericStatistics>,
    pub start_threshold_dbfs: Option<NumericStatistics>,
    pub end_threshold_dbfs: Option<NumericStatistics>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ThresholdEvaluation {
    pub threshold_ms: u64,
    pub speech_kept: usize,
    pub speech_rejected: usize,
    pub non_speech_kept: usize,
    pub non_speech_rejected: usize,
    pub speech_recall: Option<f64>,
    pub noise_rejection_rate: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct WordErrors {
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    reference_words: usize,
}

impl WordErrors {
    fn edits(self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    fn rate(self) -> Option<f64> {
        (self.reference_words > 0).then(|| self.edits() as f64 / self.reference_words as f64)
    }
}

pub(super) fn analyze(
    records: &[DatasetRecord],
    labels: &BTreeMap<String, DatasetLabel>,
) -> DatasetAnalysis {
    let samples = records
        .iter()
        .filter_map(|event| labels.get(&event.id).map(|label| (event, label)))
        .collect::<Vec<_>>();

    let detection = evaluate_detection(&samples);
    let stt = evaluate_stt(&samples);
    let historical_parser = evaluate_parser(&samples, ParserSource::Historical);
    let current_parser = evaluate_parser(&samples, ParserSource::Current);
    let speech_vad = vad_statistics(&samples, true);
    let non_speech_vad = vad_statistics(&samples, false);
    let thresholds = simulate_thresholds(&samples);
    let candidate_thresholds = select_candidate_thresholds(&thresholds);

    DatasetAnalysis {
        labeled: samples.len(),
        detection,
        stt,
        historical_parser,
        current_parser,
        speech_vad,
        non_speech_vad,
        thresholds,
        candidate_thresholds,
    }
}

pub(super) fn normalize_transcript(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    for character in text.chars().flat_map(char::to_lowercase) {
        if character.is_alphanumeric() {
            normalized.push(character);
        } else {
            normalized.push(' ');
        }
    }
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn word_errors(reference: &str, hypothesis: &str) -> WordErrors {
    let normalized_reference = normalize_transcript(reference);
    let normalized_hypothesis = normalize_transcript(hypothesis);
    let reference = normalized_reference.split_whitespace().collect::<Vec<_>>();
    let hypothesis = normalized_hypothesis.split_whitespace().collect::<Vec<_>>();
    let mut distances = vec![vec![0; hypothesis.len() + 1]; reference.len() + 1];

    for (index, row) in distances.iter_mut().enumerate() {
        row[0] = index;
    }
    for (index, value) in distances[0].iter_mut().enumerate() {
        *value = index;
    }
    for row in 1..=reference.len() {
        for column in 1..=hypothesis.len() {
            if reference[row - 1] == hypothesis[column - 1] {
                distances[row][column] = distances[row - 1][column - 1];
            } else {
                distances[row][column] = 1 + distances[row - 1][column - 1]
                    .min(distances[row - 1][column])
                    .min(distances[row][column - 1]);
            }
        }
    }

    let (mut row, mut column) = (reference.len(), hypothesis.len());
    let mut errors = WordErrors {
        reference_words: reference.len(),
        ..WordErrors::default()
    };
    while row > 0 || column > 0 {
        if row > 0
            && column > 0
            && reference[row - 1] == hypothesis[column - 1]
            && distances[row][column] == distances[row - 1][column - 1]
        {
            row -= 1;
            column -= 1;
        } else if row > 0
            && column > 0
            && distances[row][column] == distances[row - 1][column - 1] + 1
        {
            errors.substitutions += 1;
            row -= 1;
            column -= 1;
        } else if row > 0 && distances[row][column] == distances[row - 1][column] + 1 {
            errors.deletions += 1;
            row -= 1;
        } else {
            errors.insertions += 1;
            column -= 1;
        }
    }
    errors
}

fn evaluate_detection(samples: &[(&DatasetRecord, &DatasetLabel)]) -> DetectionEvaluation {
    let mut result = DetectionEvaluation::default();
    for (event, label) in samples {
        if label.actual_speech {
            result.actual_speech += 1;
            continue;
        }

        result.actual_non_speech += 1;
        if event.transcript.is_some() {
            result.non_speech_reached_stt += 1;
        }
        if event
            .transcript
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
        {
            result.non_speech_with_text += 1;
        }
        if event
            .transcript
            .as_deref()
            .is_some_and(|text| parser::parse(text).intent_name() == "no_speech")
        {
            result.non_speech_annotations += 1;
        }
        if event.transcript.is_none() && event.execution_result.status == ExecutionStatus::Rejected
        {
            result.non_speech_rejected_before_stt += 1;
        }
    }
    result
}

fn evaluate_stt(samples: &[(&DatasetRecord, &DatasetLabel)]) -> SttEvaluation {
    let mut result = SttEvaluation::default();
    let mut sample_rates = Vec::new();

    for (event, label) in samples.iter().filter(|(_, label)| label.actual_speech) {
        result.speech_samples += 1;
        let Some(reference) = label.corrected_transcript.as_deref() else {
            continue;
        };
        let errors = word_errors(reference, event.transcript.as_deref().unwrap_or(""));
        let Some(rate) = errors.rate() else {
            continue;
        };
        result.evaluated_samples += 1;
        result.samples_with_errors += usize::from(errors.edits() > 0);
        result.substitutions += errors.substitutions;
        result.deletions += errors.deletions;
        result.insertions += errors.insertions;
        result.reference_words += errors.reference_words;
        sample_rates.push(rate);
    }

    let total_edits = result.substitutions + result.deletions + result.insertions;
    result.aggregate_wer =
        (result.reference_words > 0).then(|| total_edits as f64 / result.reference_words as f64);
    result.mean_sample_wer = mean(&sample_rates);
    result.median_sample_wer = median(sample_rates);
    result
}

#[derive(Clone, Copy)]
enum ParserSource {
    Historical,
    Current,
}

fn evaluate_parser(
    samples: &[(&DatasetRecord, &DatasetLabel)],
    source: ParserSource,
) -> ParserEvaluation {
    let mut result = ParserEvaluation::default();
    for (event, label) in samples.iter().filter(|(_, label)| {
        label.actual_speech && !matches!(label.correct_intent.as_str(), "unknown" | "no_speech")
    }) {
        let predicted = match source {
            ParserSource::Historical => event
                .prediction
                .as_ref()
                .map_or("unknown", |prediction| prediction.intent.as_str()),
            ParserSource::Current => event.transcript.as_deref().map_or("unknown", |transcript| {
                parser::parse(transcript).intent_name()
            }),
        };
        result.total += 1;
        *result
            .confusion
            .entry(label.correct_intent.clone())
            .or_default()
            .entry(predicted.to_owned())
            .or_default() += 1;
        if predicted == label.correct_intent {
            result.correct += 1;
        } else {
            result.mistakes.push(ParserMistake {
                event_id: event.id.clone(),
                expected: label.correct_intent.clone(),
                predicted: predicted.to_owned(),
                transcript: event.transcript.clone().unwrap_or_default(),
            });
        }
    }
    result
}

fn vad_statistics(
    samples: &[(&DatasetRecord, &DatasetLabel)],
    actual_speech: bool,
) -> VadStatistics {
    let telemetry = samples
        .iter()
        .filter(|(_, label)| label.actual_speech == actual_speech)
        .filter_map(|(event, _)| event.vad.as_ref())
        .collect::<Vec<_>>();
    let stats = |values: Vec<f64>| NumericStatistics::from_values(values);

    VadStatistics {
        duration_ms: stats(
            telemetry
                .iter()
                .filter_map(|vad| vad.duration_ms.map(|value| value as f64))
                .collect(),
        ),
        speech_windows: stats(
            telemetry
                .iter()
                .filter_map(|vad| vad.speech_windows.map(|value| value as f64))
                .collect(),
        ),
        peak_dbfs: stats(
            telemetry
                .iter()
                .filter_map(|vad| vad.peak_dbfs.map(f64::from))
                .collect(),
        ),
        mean_speech_dbfs: stats(
            telemetry
                .iter()
                .filter_map(|vad| vad.mean_speech_dbfs.map(f64::from))
                .collect(),
        ),
        median_speech_dbfs: stats(
            telemetry
                .iter()
                .filter_map(|vad| vad.median_speech_dbfs.map(f64::from))
                .collect(),
        ),
        noise_floor_dbfs: stats(
            telemetry
                .iter()
                .filter_map(|vad| vad.noise_floor_dbfs.map(f64::from))
                .collect(),
        ),
        start_threshold_dbfs: stats(
            telemetry
                .iter()
                .filter_map(|vad| vad.start_threshold_dbfs.map(f64::from))
                .collect(),
        ),
        end_threshold_dbfs: stats(
            telemetry
                .iter()
                .filter_map(|vad| vad.end_threshold_dbfs.map(f64::from))
                .collect(),
        ),
    }
}

fn simulate_thresholds(samples: &[(&DatasetRecord, &DatasetLabel)]) -> Vec<ThresholdEvaluation> {
    let observations = samples
        .iter()
        .filter_map(|(event, label)| {
            let windows = event.vad.as_ref()?.speech_windows? as u64;
            let window_ms = event
                .provenance
                .as_ref()
                .map_or(LEGACY_LEVEL_WINDOW_MS, |provenance| {
                    provenance.vad_config.level_window_ms
                });
            Some((label.actual_speech, windows * window_ms))
        })
        .collect::<Vec<_>>();

    (40..=500)
        .step_by(20)
        .map(|threshold_ms| {
            let mut result = ThresholdEvaluation {
                threshold_ms,
                speech_kept: 0,
                speech_rejected: 0,
                non_speech_kept: 0,
                non_speech_rejected: 0,
                speech_recall: None,
                noise_rejection_rate: None,
            };
            for &(actual_speech, duration_ms) in &observations {
                match (actual_speech, duration_ms >= threshold_ms) {
                    (true, true) => result.speech_kept += 1,
                    (true, false) => result.speech_rejected += 1,
                    (false, true) => result.non_speech_kept += 1,
                    (false, false) => result.non_speech_rejected += 1,
                }
            }
            let speech_total = result.speech_kept + result.speech_rejected;
            let non_speech_total = result.non_speech_kept + result.non_speech_rejected;
            result.speech_recall =
                (speech_total > 0).then(|| result.speech_kept as f64 / speech_total as f64);
            result.noise_rejection_rate = (non_speech_total > 0)
                .then(|| result.non_speech_rejected as f64 / non_speech_total as f64);
            result
        })
        .collect()
}

fn select_candidate_thresholds(rows: &[ThresholdEvaluation]) -> Vec<u64> {
    let mut candidates = Vec::new();
    for minimum_recall in [0.98, 0.95, 0.90] {
        if let Some(row) = rows
            .iter()
            .filter(|row| {
                row.speech_recall.is_some_and(|rate| rate >= minimum_recall)
                    && row.noise_rejection_rate.is_some()
            })
            .max_by_key(|row| row.threshold_ms)
        {
            candidates.push(row.threshold_ms);
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

impl ParserEvaluation {
    pub fn accuracy(&self) -> Option<f64> {
        (self.total > 0).then(|| self.correct as f64 / self.total as f64)
    }
}

impl NumericStatistics {
    fn from_values(mut values: Vec<f64>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable_by(f64::total_cmp);
        Some(Self {
            count: values.len(),
            minimum: values[0],
            median: median_sorted(&values),
            mean: values.iter().sum::<f64>() / values.len() as f64,
            maximum: values[values.len() - 1],
        })
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    Some(median_sorted(&values))
}

fn median_sorted(values: &[f64]) -> f64 {
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{
        ExecutionResult, GroundTruth, InputMetadata, IntentPrediction, VadTelemetry,
    };

    fn event(
        id: &str,
        transcript: Option<&str>,
        predicted: Option<&str>,
        windows: usize,
    ) -> DatasetRecord {
        DatasetRecord {
            schema_version: 1,
            id: id.to_owned(),
            timestamp: "2026-09-10T00:00:00Z".to_owned(),
            provenance: None,
            audio_path: Some(format!("utterances/{id}.wav")),
            input: InputMetadata {
                device: "Test".to_owned(),
                sample_rate_hz: 16_000,
                channels: 1,
            },
            transcript: transcript.map(str::to_owned),
            prediction: predicted.map(|intent| IntentPrediction {
                intent: intent.to_owned(),
                slots: BTreeMap::new(),
            }),
            execution_result: ExecutionResult::not_attempted(),
            processing_error: None,
            vad: Some(VadTelemetry {
                speech_windows: Some(windows),
                duration_ms: Some(windows as u64 * 20),
                ..VadTelemetry::default()
            }),
            ground_truth: GroundTruth::default(),
        }
    }

    fn label(id: &str, speech: bool, transcript: Option<&str>, intent: &str) -> DatasetLabel {
        DatasetLabel {
            schema_version: 1,
            event_id: id.to_owned(),
            labeled_at: "2026-09-10T00:00:01Z".to_owned(),
            actual_speech: speech,
            corrected_transcript: transcript.map(str::to_owned),
            correct_intent: intent.to_owned(),
            notes: None,
        }
    }

    #[test]
    fn normalizes_case_punctuation_and_whitespace() {
        assert_eq!(
            normalize_transcript("  ОТКРОЙ,   Spotify!  "),
            "открой spotify"
        );
    }

    #[test]
    fn calculates_known_word_error_counts() {
        assert_eq!(
            word_errors("один два три", "один пять три снова"),
            WordErrors {
                substitutions: 1,
                insertions: 1,
                reference_words: 3,
                ..WordErrors::default()
            }
        );
        assert_eq!(word_errors("один два", "один").deletions, 1);
    }

    #[test]
    fn joins_only_events_that_have_labels() {
        let records = [
            event("labeled", Some("текст"), Some("unknown"), 10),
            event("pending", None, None, 2),
        ];
        let labels = BTreeMap::from([(
            "labeled".to_owned(),
            label("labeled", true, Some("текст"), "unknown"),
        )]);

        assert_eq!(analyze(&records, &labels).labeled, 1);
    }

    #[test]
    fn separates_historical_and_current_parser_evaluation() {
        let records = [event(
            "spotify",
            Some("Открою Spotify"),
            Some("unknown"),
            20,
        )];
        let labels = BTreeMap::from([(
            "spotify".to_owned(),
            label("spotify", true, Some("Открой Spotify"), "play_music"),
        )]);
        let analysis = analyze(&records, &labels);

        assert_eq!(analysis.historical_parser.correct, 0);
        assert_eq!(analysis.current_parser.correct, 1);
        assert_eq!(analysis.current_parser.total, 1);
    }

    #[test]
    fn simulates_minimum_speech_thresholds() {
        let records = [
            event("speech", Some("который час"), Some("tell_time"), 10),
            event("noise", None, None, 5),
        ];
        let labels = BTreeMap::from([
            (
                "speech".to_owned(),
                label("speech", true, Some("который час"), "tell_time"),
            ),
            ("noise".to_owned(), label("noise", false, None, "no_speech")),
        ]);
        let analysis = analyze(&records, &labels);
        let row = analysis
            .thresholds
            .iter()
            .find(|row| row.threshold_ms == 140)
            .unwrap();

        assert_eq!(row.speech_kept, 1);
        assert_eq!(row.non_speech_rejected, 1);
        assert_eq!(row.speech_recall, Some(1.0));
        assert_eq!(row.noise_rejection_rate, Some(1.0));
    }

    #[test]
    fn handles_empty_dataset() {
        let analysis = analyze(&[], &BTreeMap::new());

        assert_eq!(analysis.labeled, 0);
        assert_eq!(analysis.stt.aggregate_wer, None);
        assert_eq!(analysis.historical_parser.accuracy(), None);
        assert!(analysis.candidate_thresholds.is_empty());
    }
}
