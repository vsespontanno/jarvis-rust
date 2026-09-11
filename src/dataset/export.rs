use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::{DatasetLabel, DatasetRecord, analysis::normalize_transcript};

const EXPORT_SCHEMA_VERSION: u8 = 1;
const INTENTS: [&str; 4] = ["play_music", "set_timer", "tell_time", "unknown"];

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    const ALL: [Self; 3] = [Self::Train, Self::Validation, Self::Test];

    fn filename(self) -> &'static str {
        match self {
            Self::Train => "train.jsonl",
            Self::Validation => "validation.jsonl",
            Self::Test => "test.jsonl",
        }
    }
}

impl fmt::Display for Split {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Train => write!(formatter, "Train"),
            Self::Validation => write!(formatter, "Validation"),
            Self::Test => write!(formatter, "Test"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExportSource {
    Natural,
    Prompted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ExportRecord {
    schema_version: u8,
    event_id: String,
    session_id: String,
    transcript: String,
    intent: String,
    source: ExportSource,
    audio_path: String,
    split: Split,
}

#[derive(Debug, Default, PartialEq)]
struct SplitSummary {
    total: usize,
    sessions: BTreeSet<String>,
    intents: BTreeMap<String, usize>,
}

#[derive(Debug, Default)]
struct ExportReport {
    rows: usize,
    skipped_non_speech: usize,
    missing_labels: usize,
    missing_transcripts: usize,
    missing_sessions: usize,
    missing_audio: usize,
    invalid_intents: usize,
    malformed_lines: usize,
    summaries: BTreeMap<Split, SplitSummary>,
    warnings: Vec<String>,
}

pub fn export(dataset_root: &Path, output_root: &Path) -> Result<()> {
    let mut report = ExportReport::default();
    let records =
        read_jsonl::<DatasetRecord>(&dataset_root.join("events.jsonl"), "event", &mut report)?;
    let labels = read_labels(&dataset_root.join("labels.jsonl"), &mut report)?;
    let rows = build_rows(dataset_root, &records, &labels, &mut report);
    add_data_quality_warnings(&mut report);
    validate_no_session_leakage(&rows)?;
    add_duplicate_warnings(&rows, &mut report.warnings);
    report.summaries = summarize(&rows);
    add_missing_class_warnings(&report.summaries, &mut report.warnings);
    report.rows = rows.len();
    write_splits(output_root, &rows)?;

    println!("Dataset export: {}", output_root.display());
    println!("Eligible samples: {}", report.rows);
    println!("Skipped non-speech: {}", report.skipped_non_speech);
    println!("Missing labels: {}", report.missing_labels);
    println!("Missing transcripts: {}", report.missing_transcripts);
    println!("Missing session IDs: {}", report.missing_sessions);
    println!("Missing audio references/files: {}", report.missing_audio);
    println!("Invalid intents: {}", report.invalid_intents);
    println!("Malformed JSONL lines: {}", report.malformed_lines);
    for split in Split::ALL {
        print_summary(split, report.summaries.get(&split));
    }
    println!("\nWarnings:");
    if report.warnings.is_empty() {
        println!("  (none)");
    } else {
        for warning in &report.warnings {
            println!("  - {warning}");
        }
    }
    Ok(())
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(
    path: &Path,
    record_name: &str,
    report: &mut ExportReport,
) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut records = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line =
            line.with_context(|| format!("failed to read {} line {line_number}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&line) {
            Ok(record) => records.push(record),
            Err(error) => {
                report.malformed_lines += 1;
                report.warnings.push(format!(
                    "malformed {record_name} on {} line {line_number}: {error}",
                    path.display()
                ));
            }
        }
    }
    Ok(records)
}

fn read_labels(path: &Path, report: &mut ExportReport) -> Result<BTreeMap<String, DatasetLabel>> {
    if !path.exists() {
        report
            .warnings
            .push(format!("label file does not exist: {}", path.display()));
        return Ok(BTreeMap::new());
    }
    let labels = read_jsonl::<DatasetLabel>(path, "label", report)?;
    Ok(labels
        .into_iter()
        .map(|label| (label.event_id.clone(), label))
        .collect())
}

fn build_rows(
    dataset_root: &Path,
    records: &[DatasetRecord],
    labels: &BTreeMap<String, DatasetLabel>,
    report: &mut ExportReport,
) -> Vec<ExportRecord> {
    let mut rows = Vec::new();
    for event in records {
        let Some(label) = labels.get(&event.id) else {
            report.missing_labels += 1;
            continue;
        };
        if !label.actual_speech {
            report.skipped_non_speech += 1;
            continue;
        }
        let Some(transcript) = event
            .transcript
            .as_deref()
            .filter(|text| !text.trim().is_empty())
        else {
            report.missing_transcripts += 1;
            continue;
        };
        let Some(provenance) = &event.provenance else {
            report.missing_sessions += 1;
            continue;
        };
        if provenance.session_id.trim().is_empty() {
            report.missing_sessions += 1;
            continue;
        }
        if !INTENTS.contains(&label.correct_intent.as_str()) {
            report.invalid_intents += 1;
            continue;
        }
        let Some(audio_path) = event
            .audio_path
            .as_deref()
            .filter(|path| !path.trim().is_empty())
        else {
            report.missing_audio += 1;
            continue;
        };
        if !dataset_root.join(audio_path).is_file() {
            report.missing_audio += 1;
            continue;
        }

        rows.push(ExportRecord {
            schema_version: EXPORT_SCHEMA_VERSION,
            event_id: event.id.clone(),
            session_id: provenance.session_id.clone(),
            transcript: transcript.to_owned(),
            intent: label.correct_intent.clone(),
            source: if event.collection.is_some() {
                ExportSource::Prompted
            } else {
                ExportSource::Natural
            },
            audio_path: audio_path.to_owned(),
            split: Split::Train,
        });
    }
    let assignments = assign_sessions(rows.iter().map(|row| row.session_id.as_str()));
    for row in &mut rows {
        row.split = assignments[&row.session_id];
    }
    rows
}

fn stable_hash(session_id: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in session_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn assign_sessions<'a>(session_ids: impl Iterator<Item = &'a str>) -> BTreeMap<String, Split> {
    let mut sessions = session_ids
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    sessions.sort_by_key(|session| (stable_hash(session), session.clone()));
    let total = sessions.len();
    let held_out = if total >= 3 {
        ((total as f64 * 0.15).round() as usize).max(1)
    } else {
        0
    };
    let validation_start = total.saturating_sub(held_out * 2);
    let test_start = total.saturating_sub(held_out);

    sessions
        .into_iter()
        .enumerate()
        .map(|(index, session)| {
            let split = if index >= test_start {
                Split::Test
            } else if index >= validation_start {
                Split::Validation
            } else {
                Split::Train
            };
            (session, split)
        })
        .collect()
}

fn validate_no_session_leakage(rows: &[ExportRecord]) -> Result<()> {
    let mut sessions = BTreeMap::new();
    for row in rows {
        if let Some(previous) = sessions.insert(row.session_id.as_str(), row.split)
            && previous != row.split
        {
            bail!(
                "session '{}' appears in both {} and {} splits",
                row.session_id,
                previous,
                row.split
            );
        }
    }
    Ok(())
}

fn summarize(rows: &[ExportRecord]) -> BTreeMap<Split, SplitSummary> {
    let mut summaries = Split::ALL
        .into_iter()
        .map(|split| (split, SplitSummary::default()))
        .collect::<BTreeMap<_, _>>();
    for row in rows {
        let summary = summaries.get_mut(&row.split).expect("all splits exist");
        summary.total += 1;
        summary.sessions.insert(row.session_id.clone());
        *summary.intents.entry(row.intent.clone()).or_default() += 1;
    }
    summaries
}

fn add_missing_class_warnings(
    summaries: &BTreeMap<Split, SplitSummary>,
    warnings: &mut Vec<String>,
) {
    for split in Split::ALL {
        let summary = summaries.get(&split).expect("all splits exist");
        for intent in INTENTS {
            if !summary.intents.contains_key(intent) {
                warnings.push(format!("{split} split is missing intent '{intent}'"));
            }
        }
    }
}

fn add_data_quality_warnings(report: &mut ExportReport) {
    for (count, problem) in [
        (report.missing_labels, "missing label"),
        (report.missing_transcripts, "missing transcript"),
        (report.missing_sessions, "missing session_id"),
        (report.missing_audio, "missing audio reference or file"),
        (report.invalid_intents, "invalid intent"),
    ] {
        if count > 0 {
            report
                .warnings
                .push(format!("{count} event(s) skipped: {problem}"));
        }
    }
}

fn add_duplicate_warnings(rows: &[ExportRecord], warnings: &mut Vec<String>) {
    let mut phrases: BTreeMap<String, BTreeSet<Split>> = BTreeMap::new();
    for row in rows {
        let normalized = normalize_transcript(&row.transcript);
        if !normalized.is_empty() {
            phrases.entry(normalized).or_default().insert(row.split);
        }
    }
    for (transcript, splits) in phrases {
        if splits.len() > 1 {
            let names = splits
                .into_iter()
                .map(|split| split.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            warnings.push(format!(
                "normalized transcript '{transcript}' occurs across splits: {names}"
            ));
        }
    }
}

fn write_splits(output_root: &Path, rows: &[ExportRecord]) -> Result<()> {
    fs::create_dir_all(output_root)
        .with_context(|| format!("failed to create {}", output_root.display()))?;
    for split in Split::ALL {
        let path = output_root.join(split.filename());
        let file = File::create(&path)
            .with_context(|| format!("failed to create export file {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        for row in rows.iter().filter(|row| row.split == split) {
            serde_json::to_writer(&mut writer, row).context("failed to serialize export row")?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
    }
    Ok(())
}

fn print_summary(split: Split, summary: Option<&SplitSummary>) {
    println!("\n{split}:");
    let Some(summary) = summary else {
        println!("  Total: 0\n  Sessions: 0");
        return;
    };
    println!("  Total: {}", summary.total);
    println!("  Sessions: {}", summary.sessions.len());
    for intent in INTENTS {
        println!(
            "  {intent}: {}",
            summary.intents.get(intent).copied().unwrap_or(0)
        );
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::dataset::{
        ExecutionResult, GroundTruth, InputMetadata, IntentPrediction, Provenance,
        VadConfigMetadata,
    };

    fn test_directory() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "jarvis-export-test-{}-{unique}",
            std::process::id()
        ))
    }

    fn record(id: &str, session: Option<&str>, transcript: Option<&str>) -> DatasetRecord {
        DatasetRecord {
            schema_version: 3,
            id: id.to_owned(),
            timestamp: "2026-09-11T00:00:00Z".to_owned(),
            provenance: session.map(|session_id| Provenance {
                jarvis_version: "0.6.0".to_owned(),
                session_id: session_id.to_owned(),
                whisper_model: "models/test.bin".to_owned(),
                parser_version: 2,
                vad_config: VadConfigMetadata {
                    level_window_ms: 20,
                    calibration_ms: 1_000,
                    speech_start_ms: 60,
                    speech_end_ms: 600,
                    min_speech_duration_ms: 140,
                    pre_roll_ms: 300,
                    start_margin_db: 12.0,
                    end_margin_db: 6.0,
                    minimum_start_level_dbfs: -35.0,
                    minimum_end_level_dbfs: -40.0,
                },
            }),
            collection: None,
            audio_path: Some(format!("utterances/{id}.wav")),
            input: InputMetadata {
                device: "Test microphone".to_owned(),
                sample_rate_hz: 16_000,
                channels: 1,
            },
            transcript: transcript.map(str::to_owned),
            prediction: Some(IntentPrediction {
                intent: "unknown".to_owned(),
                slots: BTreeMap::new(),
            }),
            execution_result: ExecutionResult::not_attempted(),
            processing_error: None,
            vad: None,
            ground_truth: GroundTruth::default(),
        }
    }

    fn label(id: &str, intent: &str) -> DatasetLabel {
        DatasetLabel {
            schema_version: 1,
            event_id: id.to_owned(),
            labeled_at: "2026-09-11T00:00:01Z".to_owned(),
            actual_speech: true,
            corrected_transcript: Some("reference".to_owned()),
            correct_intent: intent.to_owned(),
            notes: None,
        }
    }

    fn row(session: &str, transcript: &str, intent: &str, split: Split) -> ExportRecord {
        ExportRecord {
            schema_version: 1,
            event_id: format!("{session}-{intent}"),
            session_id: session.to_owned(),
            transcript: transcript.to_owned(),
            intent: intent.to_owned(),
            source: ExportSource::Natural,
            audio_path: "utterances/test.wav".to_owned(),
            split,
        }
    }

    #[test]
    fn split_is_deterministic_and_session_scoped() {
        let sessions = ["one", "two", "three", "four", "five"];
        let first = assign_sessions(sessions.into_iter());
        let second = assign_sessions(sessions.into_iter().rev());

        assert_eq!(first, second);
        assert_eq!(
            first
                .values()
                .filter(|&&split| split == Split::Train)
                .count(),
            3
        );
        assert_eq!(
            first
                .values()
                .filter(|&&split| split == Split::Validation)
                .count(),
            1
        );
        assert_eq!(
            first
                .values()
                .filter(|&&split| split == Split::Test)
                .count(),
            1
        );
    }

    #[test]
    fn rejects_session_leakage() {
        let rows = [
            row("same", "который час", "tell_time", Split::Train),
            row("same", "включи музыку", "play_music", Split::Test),
        ];

        assert!(
            validate_no_session_leakage(&rows)
                .unwrap_err()
                .to_string()
                .contains("both Train and Test")
        );
    }

    #[test]
    fn reports_class_counts() {
        let rows = [
            row("one", "который час", "tell_time", Split::Train),
            row("two", "сколько времени", "tell_time", Split::Train),
            row("one", "включи музыку", "play_music", Split::Train),
        ];

        let summaries = summarize(&rows);
        let train = &summaries[&Split::Train];
        assert_eq!(train.total, 3);
        assert_eq!(train.sessions.len(), 2);
        assert_eq!(train.intents["tell_time"], 2);
        assert_eq!(train.intents["play_music"], 1);
    }

    #[test]
    fn warns_about_normalized_transcript_across_splits() {
        let rows = [
            row("one", "Открой, Spotify!", "play_music", Split::Train),
            row("two", "открой spotify", "play_music", Split::Test),
        ];
        let mut warnings = Vec::new();

        add_duplicate_warnings(&rows, &mut warnings);

        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("открой spotify"));
    }

    #[test]
    fn warns_and_skips_incomplete_events() {
        let root = test_directory();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("events.jsonl"),
            format!(
                "not-json\n{}\n{}\n",
                serde_json::to_string(&record("missing-label", Some("s1"), Some("text"))).unwrap(),
                serde_json::to_string(&record("missing-session", None, Some("text"))).unwrap()
            ),
        )
        .unwrap();
        fs::write(
            root.join("labels.jsonl"),
            format!(
                "{}\n",
                serde_json::to_string(&label("missing-session", "unknown")).unwrap()
            ),
        )
        .unwrap();
        let mut report = ExportReport::default();

        let records =
            read_jsonl::<DatasetRecord>(&root.join("events.jsonl"), "event", &mut report).unwrap();
        let labels = read_labels(&root.join("labels.jsonl"), &mut report).unwrap();
        let rows = build_rows(&root, &records, &labels, &mut report);
        add_data_quality_warnings(&mut report);

        assert!(rows.is_empty());
        assert_eq!(report.malformed_lines, 1);
        assert_eq!(report.missing_labels, 1);
        assert_eq!(report.missing_sessions, 1);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("missing label"))
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("missing session_id"))
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repeated_export_writes_identical_manifests() {
        let root = test_directory();
        let output = root.join("splits");
        fs::create_dir_all(root.join("utterances")).unwrap();
        let event = record("event", Some("stable-session"), Some("который час"));
        fs::write(root.join("utterances/event.wav"), b"wav").unwrap();
        fs::write(
            root.join("events.jsonl"),
            format!("{}\n", serde_json::to_string(&event).unwrap()),
        )
        .unwrap();
        fs::write(
            root.join("labels.jsonl"),
            format!(
                "{}\n",
                serde_json::to_string(&label("event", "tell_time")).unwrap()
            ),
        )
        .unwrap();

        export(&root, &output).unwrap();
        let first = Split::ALL.map(|split| fs::read(output.join(split.filename())).unwrap());
        export(&root, &output).unwrap();
        let second = Split::ALL.map(|split| fs::read(output.join(split.filename())).unwrap());

        assert_eq!(first, second);
        fs::remove_dir_all(root).unwrap();
    }
}
