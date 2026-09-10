use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};

use crate::audio::Recording;

mod report;
mod review;

pub use report::print_report;
pub use review::review;

const SCHEMA_VERSION: u8 = 2;
const LABEL_SCHEMA_VERSION: u8 = 1;
static ID_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub struct DatasetStore {
    root: PathBuf,
    events_path: PathBuf,
    provenance: Provenance,
}

#[derive(Debug)]
pub struct SampleDescriptor {
    pub id: String,
    pub timestamp: String,
    pub audio_path: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetRecord {
    pub schema_version: u8,
    pub id: String,
    pub timestamp: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    pub audio_path: Option<String>,
    pub input: InputMetadata,
    pub transcript: Option<String>,
    pub prediction: Option<IntentPrediction>,
    pub execution_result: ExecutionResult,
    pub processing_error: Option<String>,
    pub vad: Option<VadTelemetry>,
    #[serde(default, skip_serializing_if = "GroundTruth::is_empty")]
    pub ground_truth: GroundTruth,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Provenance {
    pub jarvis_version: String,
    pub session_id: String,
    pub whisper_model: String,
    pub parser_version: u32,
    pub vad_config: VadConfigMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VadConfigMetadata {
    pub level_window_ms: u64,
    pub calibration_ms: u64,
    pub speech_start_ms: u64,
    pub speech_end_ms: u64,
    pub min_speech_duration_ms: u64,
    pub pre_roll_ms: u64,
    pub start_margin_db: f32,
    pub end_margin_db: f32,
    pub minimum_start_level_dbfs: f32,
    pub minimum_end_level_dbfs: f32,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct InputMetadata {
    pub device: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct IntentPrediction {
    pub intent: String,
    pub slots: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ExecutionResult {
    pub status: ExecutionStatus,
    pub response: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    NotAttempted,
    Rejected,
    Succeeded,
    Unsupported,
    Failed,
}

#[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct VadTelemetry {
    pub noise_floor_dbfs: Option<f32>,
    pub start_threshold_dbfs: Option<f32>,
    pub end_threshold_dbfs: Option<f32>,
    pub peak_dbfs: Option<f32>,
    pub mean_speech_dbfs: Option<f32>,
    pub median_speech_dbfs: Option<f32>,
    pub duration_ms: Option<u64>,
    pub trailing_silence_ms: Option<u64>,
    pub speech_windows: Option<usize>,
    pub silence_windows: Option<usize>,
    pub end_reason: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GroundTruth {
    pub corrected_transcript: Option<String>,
    pub correct_intent: Option<String>,
    pub actual_speech: Option<bool>,
    pub notes: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetLabel {
    pub schema_version: u8,
    pub event_id: String,
    pub labeled_at: String,
    pub actual_speech: bool,
    pub corrected_transcript: Option<String>,
    pub correct_intent: String,
    pub notes: Option<String>,
}

pub struct LabelStore {
    path: PathBuf,
}

impl DatasetStore {
    pub fn open(root: impl AsRef<Path>, provenance: Provenance) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let utterances = root.join("utterances");
        fs::create_dir_all(&utterances).with_context(|| {
            format!(
                "failed to create the dataset directory {}",
                utterances.display()
            )
        })?;

        Ok(Self {
            events_path: root.join("events.jsonl"),
            root,
            provenance,
        })
    }

    pub fn new_sample(&self) -> SampleDescriptor {
        let timestamp = Timestamp::now().to_string();
        let filename_time = timestamp.replace([':', '.'], "-");
        let sequence = ID_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let id = format!("{filename_time}-{}-{sequence}", std::process::id());

        SampleDescriptor {
            audio_path: format!("utterances/{id}.wav"),
            id,
            timestamp,
        }
    }

    pub fn save_audio(&self, sample: &SampleDescriptor, recording: &Recording) -> Result<()> {
        let path = self.root.join(&sample.audio_path);
        recording
            .write_wav(&path)
            .with_context(|| format!("failed to save dataset audio to {}", path.display()))
    }

    pub fn append(&self, record: &DatasetRecord) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.events_path)
            .with_context(|| format!("failed to open {}", self.events_path.display()))?;

        serde_json::to_writer(&mut file, record).context("failed to serialize dataset record")?;
        file.write_all(b"\n")
            .context("failed to finish the JSON Lines record")?;
        file.flush().context("failed to flush the dataset record")?;
        Ok(())
    }

    pub fn new_record(&self, sample: &SampleDescriptor, recording: &Recording) -> DatasetRecord {
        DatasetRecord::new(sample, recording, self.provenance.clone())
    }
}

impl DatasetRecord {
    fn new(sample: &SampleDescriptor, recording: &Recording, provenance: Provenance) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            id: sample.id.clone(),
            timestamp: sample.timestamp.clone(),
            provenance: Some(provenance),
            audio_path: Some(sample.audio_path.clone()),
            input: InputMetadata {
                device: recording.device_name.clone(),
                sample_rate_hz: recording.sample_rate,
                channels: recording.channels,
            },
            transcript: None,
            prediction: None,
            execution_result: ExecutionResult::not_attempted(),
            processing_error: None,
            vad: None,
            ground_truth: GroundTruth::default(),
        }
    }
}

impl Provenance {
    pub fn new_session_id() -> String {
        let timestamp = Timestamp::now().to_string().replace([':', '.'], "-");
        format!("session-{timestamp}-{}", std::process::id())
    }
}

impl GroundTruth {
    fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

impl DatasetLabel {
    pub fn new(
        event_id: String,
        actual_speech: bool,
        corrected_transcript: Option<String>,
        correct_intent: String,
        notes: Option<String>,
    ) -> Self {
        Self {
            schema_version: LABEL_SCHEMA_VERSION,
            event_id,
            labeled_at: Timestamp::now().to_string(),
            actual_speech,
            corrected_transcript,
            correct_intent,
            notes,
        }
    }
}

impl LabelStore {
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root)
            .with_context(|| format!("failed to create dataset directory {}", root.display()))?;
        Ok(Self {
            path: root.join("labels.jsonl"),
        })
    }

    pub fn append(&self, label: &DatasetLabel) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("failed to open {}", self.path.display()))?;
        serde_json::to_writer(&mut file, label).context("failed to serialize dataset label")?;
        file.write_all(b"\n")
            .context("failed to finish the label JSON Lines record")?;
        file.flush().context("failed to flush the dataset label")?;
        Ok(())
    }

    pub fn latest(&self) -> Result<BTreeMap<String, DatasetLabel>> {
        let file = match fs::File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(BTreeMap::new());
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to open {}", self.path.display()));
            }
        };
        let reader = std::io::BufReader::new(file);
        let mut labels = BTreeMap::new();

        for (index, line) in std::io::BufRead::lines(reader).enumerate() {
            let line_number = index + 1;
            let line = line.with_context(|| format!("failed to read label line {line_number}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let label: DatasetLabel = serde_json::from_str(&line)
                .with_context(|| format!("invalid label on JSONL line {line_number}"))?;
            labels.insert(label.event_id.clone(), label);
        }

        Ok(labels)
    }
}

impl ExecutionResult {
    pub fn not_attempted() -> Self {
        Self {
            status: ExecutionStatus::NotAttempted,
            response: None,
            error: None,
        }
    }
}

impl ExecutionStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::NotAttempted => "not_attempted",
            Self::Rejected => "rejected",
            Self::Succeeded => "succeeded",
            Self::Unsupported => "unsupported",
            Self::Failed => "failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use super::*;

    fn test_directory() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "jarvis-dataset-test-{}-{unique}",
            std::process::id()
        ))
    }

    fn recording() -> Recording {
        Recording {
            samples: vec![0, 1, -1],
            sample_rate: 24_000,
            channels: 1,
            device_name: "Test microphone".to_owned(),
        }
    }

    fn provenance() -> Provenance {
        Provenance {
            jarvis_version: "0.3.0".to_owned(),
            session_id: "session-test".to_owned(),
            whisper_model: "models/test.bin".to_owned(),
            parser_version: 1,
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
        }
    }

    #[test]
    fn serializes_and_deserializes_a_record() {
        let sample = SampleDescriptor {
            id: "sample-1".to_owned(),
            timestamp: "2026-09-08T12:00:00Z".to_owned(),
            audio_path: "utterances/sample-1.wav".to_owned(),
        };
        let mut record = DatasetRecord::new(&sample, &recording(), provenance());
        record.transcript = Some("который час".to_owned());
        record.prediction = Some(IntentPrediction {
            intent: "tell_time".to_owned(),
            slots: BTreeMap::new(),
        });
        record.vad = Some(VadTelemetry {
            noise_floor_dbfs: Some(-50.0),
            start_threshold_dbfs: Some(-38.0),
            end_threshold_dbfs: Some(-44.0),
            peak_dbfs: Some(-18.0),
            mean_speech_dbfs: Some(-25.0),
            median_speech_dbfs: Some(-24.0),
            duration_ms: Some(1_200),
            trailing_silence_ms: Some(600),
            speech_windows: Some(30),
            silence_windows: Some(30),
            end_reason: Some("silence".to_owned()),
        });

        let json = serde_json::to_string(&record).unwrap();
        let decoded: DatasetRecord = serde_json::from_str(&json).unwrap();

        assert!(json.contains("\"peak_dbfs\":-18.0"));
        assert!(json.contains("\"jarvis_version\":\"0.3.0\""));
        assert!(!json.contains("ground_truth"));
        assert_eq!(decoded, record);
    }

    #[test]
    fn stores_separate_audio_and_appends_json_lines() {
        let root = test_directory();
        let store = DatasetStore::open(&root, provenance()).unwrap();
        let recording = recording();

        for _ in 0..2 {
            let sample = store.new_sample();
            store.save_audio(&sample, &recording).unwrap();
            let record = store.new_record(&sample, &recording);
            store.append(&record).unwrap();
        }

        let events = fs::read_to_string(root.join("events.jsonl")).unwrap();
        assert_eq!(events.lines().count(), 2);
        assert_eq!(fs::read_dir(root.join("utterances")).unwrap().count(), 2);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn logs_rejected_detection_without_writing_audio() {
        let root = test_directory();
        let store = DatasetStore::open(&root, provenance()).unwrap();
        let sample = store.new_sample();
        let mut record = store.new_record(&sample, &recording());
        record.audio_path = None;
        record.execution_result = ExecutionResult {
            status: ExecutionStatus::Rejected,
            response: None,
            error: None,
        };

        store.append(&record).unwrap();

        let events = fs::read_to_string(root.join("events.jsonl")).unwrap();
        let decoded: DatasetRecord = serde_json::from_str(events.trim()).unwrap();
        assert_eq!(decoded.audio_path, None);
        assert_eq!(decoded.execution_result.status, ExecutionStatus::Rejected);
        assert_eq!(fs::read_dir(root.join("utterances")).unwrap().count(), 0);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn appends_labels_and_uses_latest_label_for_an_event() {
        let root = test_directory();
        let store = LabelStore::open(&root).unwrap();
        let first = DatasetLabel::new(
            "event-1".to_owned(),
            true,
            Some("старый текст".to_owned()),
            "unknown".to_owned(),
            None,
        );
        let corrected = DatasetLabel::new(
            "event-1".to_owned(),
            true,
            Some("который час".to_owned()),
            "tell_time".to_owned(),
            Some("исправлено".to_owned()),
        );

        store.append(&first).unwrap();
        store.append(&corrected).unwrap();

        let latest = store.latest().unwrap();
        assert_eq!(latest.len(), 1);
        assert_eq!(latest["event-1"], corrected);
        assert_eq!(
            fs::read_to_string(root.join("labels.jsonl"))
                .unwrap()
                .lines()
                .count(),
            2
        );

        fs::remove_dir_all(root).unwrap();
    }
}
