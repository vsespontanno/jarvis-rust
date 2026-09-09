use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result};

use super::DatasetRecord;

#[derive(Debug, PartialEq)]
struct DatasetReport {
    events: usize,
    audio_samples: usize,
    transcripts: usize,
    processing_errors: usize,
    short_rejections: usize,
    non_speech_annotations: usize,
    execution_statuses: BTreeMap<String, usize>,
    intents: BTreeMap<String, usize>,
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
    let report = DatasetReport::from_reader(BufReader::new(file))?;
    println!("Dataset: {}\n{report}", path.display());
    Ok(())
}

impl DatasetReport {
    fn from_reader(reader: impl BufRead) -> Result<Self> {
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

        Ok(Self::from_records(&records))
    }

    fn from_records(records: &[DatasetRecord]) -> Self {
        let mut execution_statuses = BTreeMap::new();
        let mut intents = BTreeMap::new();
        let mut devices = BTreeMap::new();
        let mut end_reasons = BTreeMap::new();
        let mut durations_ms = Vec::new();
        let mut speech_windows = Vec::new();
        let mut noise_floor_dbfs = Vec::new();

        for record in records {
            increment(
                &mut execution_statuses,
                record.execution_result.status.as_str(),
            );
            increment(&mut devices, record.input.device.trim());

            if let Some(prediction) = &record.prediction {
                increment(&mut intents, &prediction.intent);
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
            execution_statuses,
            intents,
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
        write_counts(formatter, "Execution", &self.execution_statuses)?;
        write_counts(formatter, "Intents", &self.intents)?;
        write_counts(formatter, "Input devices", &self.devices)?;
        write_counts(formatter, "VAD end reasons", &self.end_reasons)?;

        writeln!(formatter, "\nVAD distributions (min / median / p95 / max):")?;
        write_distribution(formatter, "duration", self.durations_ms, "ms")?;
        write_distribution(formatter, "speech windows", self.speech_windows, "")?;
        write_distribution(formatter, "noise floor", self.noise_floor_dbfs, "dBFS")
    }
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

        let report = DatasetReport::from_reader(Cursor::new(jsonl)).unwrap();

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
    fn reports_invalid_line_number() {
        let error = DatasetReport::from_reader(Cursor::new("{}\nnot-json\n")).unwrap_err();

        assert!(error.to_string().contains("JSONL line 1"));
    }
}
