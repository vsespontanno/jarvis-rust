mod features;
mod logistic;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use self::{
    features::{FeatureConfig, TfidfVectorizer},
    logistic::{LogisticRegression, TrainingConfig},
};
use crate::parser;

const MODEL_SCHEMA_VERSION: u8 = 1;
const SUPPORTED_INTENTS: [&str; 4] = ["play_music", "set_timer", "tell_time", "unknown"];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct IntentModelArtifact {
    schema_version: u8,
    feature_extractor: TfidfVectorizer,
    classifier: LogisticRegression,
    training: TrainingMetadata,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TrainingMetadata {
    samples: usize,
    sessions: Vec<String>,
    class_counts: BTreeMap<String, usize>,
    feature_config: FeatureConfig,
    optimization: TrainingConfig,
    initial_loss: f64,
    final_loss: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExperimentSample {
    event_id: String,
    session_id: String,
    transcript: String,
    intent: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationSplit {
    Validation,
    Test,
}

impl EvaluationSplit {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "validation" => Ok(Self::Validation),
            "test" => Ok(Self::Test),
            _ => bail!("evaluation split must be 'validation' or 'test'"),
        }
    }

    fn filename(self) -> &'static str {
        match self {
            Self::Validation => "validation.jsonl",
            Self::Test => "test.jsonl",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct EvaluationMetrics {
    correct: usize,
    total: usize,
    supported_correct: usize,
    supported_total: usize,
    false_commands: usize,
    negatives: usize,
    confusion: BTreeMap<String, BTreeMap<String, usize>>,
}

#[derive(Clone, Copy)]
enum Strategy {
    Rules,
    MachineLearning,
    Hybrid,
}

impl Strategy {
    const ALL: [Self; 3] = [Self::Rules, Self::MachineLearning, Self::Hybrid];

    fn name(self) -> &'static str {
        match self {
            Self::Rules => "Rule parser",
            Self::MachineLearning => "ML classifier",
            Self::Hybrid => "Agreement-gated hybrid",
        }
    }
}

pub fn train(splits_root: &Path, model_path: &Path) -> Result<()> {
    let samples = read_samples(&splits_root.join("train.jsonl"))?;
    validate_samples(&samples, "train")?;
    let sessions = sample_sessions(&samples);
    if sessions.len() < 3 {
        eprintln!(
            "Warning: train contains only {} session(s); this model is a smoke-test artifact, not a trustworthy result.",
            sessions.len()
        );
    }

    let texts = samples
        .iter()
        .map(|sample| sample.transcript.clone())
        .collect::<Vec<_>>();
    let targets = samples
        .iter()
        .map(|sample| sample.intent.clone())
        .collect::<Vec<_>>();
    let feature_config = FeatureConfig::default();
    let feature_extractor = TfidfVectorizer::fit(&texts, feature_config);
    ensure!(
        feature_extractor.dimensions() > 0,
        "training produced an empty feature vocabulary"
    );
    let features = texts
        .iter()
        .map(|text| feature_extractor.transform(text))
        .collect::<Vec<_>>();
    let optimization = TrainingConfig::default();
    let (classifier, stats) = LogisticRegression::train(
        &features,
        &targets,
        feature_extractor.dimensions(),
        optimization,
    )?;
    let artifact = IntentModelArtifact {
        schema_version: MODEL_SCHEMA_VERSION,
        feature_extractor,
        classifier,
        training: TrainingMetadata {
            samples: samples.len(),
            sessions: sessions.into_iter().collect(),
            class_counts: class_counts(&samples),
            feature_config,
            optimization,
            initial_loss: stats.initial_loss,
            final_loss: stats.final_loss,
        },
    };
    artifact.validate()?;
    write_artifact(model_path, &artifact)?;

    println!("Intent model: {}", model_path.display());
    println!("Training samples: {}", artifact.training.samples);
    println!("Training sessions: {}", artifact.training.sessions.len());
    println!("Classes:");
    for intent in SUPPORTED_INTENTS {
        println!(
            "  {intent}: {}",
            artifact
                .training
                .class_counts
                .get(intent)
                .copied()
                .unwrap_or(0)
        );
    }
    println!("Character n-grams: 2–5");
    println!(
        "TF-IDF features: {}",
        artifact.feature_extractor.dimensions()
    );
    println!(
        "Cross-entropy: {:.6} → {:.6}",
        stats.initial_loss, stats.final_loss
    );
    println!(
        "Safe-reject threshold: {:.2}",
        optimization.reject_threshold
    );
    Ok(())
}

pub fn evaluate(splits_root: &Path, model_path: &Path, split: EvaluationSplit) -> Result<()> {
    let artifact = read_artifact(model_path)?;
    let samples = read_samples(&splits_root.join(split.filename()))?;
    validate_samples(&samples, split.name())?;
    ensure_independent_sessions(&artifact.training.sessions, &samples)?;

    println!("Intent evaluation: {}", split.name());
    println!("Samples: {}", samples.len());
    println!("Sessions: {}", sample_sessions(&samples).len());
    for strategy in Strategy::ALL {
        let metrics = evaluate_strategy(&artifact, &samples, strategy);
        print_metrics(strategy.name(), &metrics);
    }
    Ok(())
}

impl IntentModelArtifact {
    fn predict(&self, transcript: &str) -> logistic::Prediction {
        self.classifier
            .predict(&self.feature_extractor.transform(transcript))
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == MODEL_SCHEMA_VERSION,
            "unsupported intent model schema {}",
            self.schema_version
        );
        let dimensions = self.feature_extractor.dimensions();
        let indices = self
            .feature_extractor
            .vocabulary
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        let expected_indices = (0..dimensions).collect::<BTreeSet<_>>();
        ensure!(
            indices == expected_indices,
            "TF-IDF vocabulary indices are not contiguous"
        );
        ensure!(
            self.feature_extractor.inverse_document_frequency.len() == dimensions,
            "TF-IDF dimensions do not match vocabulary"
        );
        ensure!(
            self.feature_extractor
                .inverse_document_frequency
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "intent model contains invalid IDF values"
        );
        ensure!(
            self.classifier.labels.len() == self.classifier.weights.len()
                && self.classifier.labels.len() == self.classifier.biases.len(),
            "classifier class dimensions do not match"
        );
        ensure!(
            self.classifier
                .weights
                .iter()
                .all(|weights| weights.len() == dimensions),
            "classifier feature dimensions do not match TF-IDF vocabulary"
        );
        ensure!(
            self.classifier
                .weights
                .iter()
                .flatten()
                .chain(&self.classifier.biases)
                .all(|value| value.is_finite()),
            "intent model contains non-finite parameters"
        );
        ensure!(
            (0.0..=1.0).contains(&self.classifier.reject_threshold),
            "intent model has an invalid reject threshold"
        );
        ensure!(
            self.classifier
                .labels
                .iter()
                .any(|label| label == "unknown"),
            "intent model has no unknown class"
        );
        Ok(())
    }
}

fn read_samples(path: &Path) -> Result<Vec<ExperimentSample>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut samples = Vec::new();
    let mut event_ids = BTreeSet::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line =
            line.with_context(|| format!("failed to read {} line {line_number}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let sample: ExperimentSample = serde_json::from_str(&line)
            .with_context(|| format!("invalid sample on {} line {line_number}", path.display()))?;
        ensure!(
            event_ids.insert(sample.event_id.clone()),
            "duplicate event_id '{}' in {}",
            sample.event_id,
            path.display()
        );
        samples.push(sample);
    }
    Ok(samples)
}

fn validate_samples(samples: &[ExperimentSample], split: &str) -> Result<()> {
    ensure!(!samples.is_empty(), "{split} split is empty");
    for sample in samples {
        ensure!(
            !sample.session_id.trim().is_empty(),
            "event '{}' has no session_id",
            sample.event_id
        );
        ensure!(
            !sample.transcript.trim().is_empty(),
            "event '{}' has no transcript",
            sample.event_id
        );
        ensure!(
            SUPPORTED_INTENTS.contains(&sample.intent.as_str()),
            "event '{}' has unsupported intent '{}'",
            sample.event_id,
            sample.intent
        );
    }
    if split == "train" {
        let present = samples
            .iter()
            .map(|sample| sample.intent.as_str())
            .collect::<BTreeSet<_>>();
        for intent in SUPPORTED_INTENTS {
            ensure!(
                present.contains(intent),
                "train split is missing intent '{intent}'"
            );
        }
    }
    Ok(())
}

fn sample_sessions(samples: &[ExperimentSample]) -> BTreeSet<String> {
    samples
        .iter()
        .map(|sample| sample.session_id.clone())
        .collect()
}

fn class_counts(samples: &[ExperimentSample]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        *counts.entry(sample.intent.clone()).or_default() += 1;
    }
    counts
}

fn ensure_independent_sessions(
    training_sessions: &[String],
    evaluation_samples: &[ExperimentSample],
) -> Result<()> {
    let training = training_sessions.iter().collect::<BTreeSet<_>>();
    let evaluation = evaluation_samples
        .iter()
        .map(|sample| &sample.session_id)
        .collect::<BTreeSet<_>>();
    let overlap = training
        .intersection(&evaluation)
        .cloned()
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        overlap.is_empty(),
        "session leakage between training and evaluation: {}",
        overlap.join(", ")
    );
    Ok(())
}

fn evaluate_strategy(
    artifact: &IntentModelArtifact,
    samples: &[ExperimentSample],
    strategy: Strategy,
) -> EvaluationMetrics {
    let mut metrics = EvaluationMetrics::default();
    for sample in samples {
        let rule_intent = parser::parse(&sample.transcript).intent_name().to_owned();
        let ml_intent = artifact.predict(&sample.transcript).intent;
        let prediction = match strategy {
            Strategy::Rules => rule_intent,
            Strategy::MachineLearning => ml_intent,
            Strategy::Hybrid => hybrid_prediction(&rule_intent, &ml_intent).to_owned(),
        };
        metrics.total += 1;
        metrics.correct += usize::from(prediction == sample.intent);
        if sample.intent == "unknown" {
            metrics.negatives += 1;
            metrics.false_commands += usize::from(prediction != "unknown");
        } else {
            metrics.supported_total += 1;
            metrics.supported_correct += usize::from(prediction == sample.intent);
        }
        *metrics
            .confusion
            .entry(sample.intent.clone())
            .or_default()
            .entry(prediction)
            .or_default() += 1;
    }
    metrics
}

fn hybrid_prediction<'a>(rule_intent: &'a str, ml_intent: &'a str) -> &'a str {
    if rule_intent == "unknown" {
        ml_intent
    } else if rule_intent == ml_intent {
        rule_intent
    } else {
        "unknown"
    }
}

fn print_metrics(name: &str, metrics: &EvaluationMetrics) {
    println!("\n{name}:");
    println!("  Accuracy: {}", ratio(metrics.correct, metrics.total));
    println!(
        "  Supported-command accuracy: {}",
        ratio(metrics.supported_correct, metrics.supported_total)
    );
    println!(
        "  False commands on negatives: {}",
        ratio(metrics.false_commands, metrics.negatives)
    );
    println!("  Per-class precision / recall / F1:");
    for intent in SUPPORTED_INTENTS {
        let true_positive = confusion_count(&metrics.confusion, intent, intent);
        let predicted = metrics
            .confusion
            .values()
            .map(|row| row.get(intent).copied().unwrap_or(0))
            .sum::<usize>();
        let actual = metrics
            .confusion
            .get(intent)
            .map(|row| row.values().sum())
            .unwrap_or(0);
        let precision = fraction(true_positive, predicted);
        let recall = fraction(true_positive, actual);
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        println!("    {intent}: {precision:.3} / {recall:.3} / {f1:.3}");
    }
    println!("  Confusion (actual -> predicted):");
    for actual in SUPPORTED_INTENTS {
        let cells = SUPPORTED_INTENTS
            .iter()
            .map(|predicted| {
                format!(
                    "{predicted}={}",
                    confusion_count(&metrics.confusion, actual, predicted)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("    {actual}: {cells}");
    }
}

fn confusion_count(
    confusion: &BTreeMap<String, BTreeMap<String, usize>>,
    actual: &str,
    predicted: &str,
) -> usize {
    confusion
        .get(actual)
        .and_then(|row| row.get(predicted))
        .copied()
        .unwrap_or(0)
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn ratio(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        return "n/a".to_owned();
    }
    format!(
        "{numerator}/{denominator} ({:.1}%)",
        100.0 * fraction(numerator, denominator)
    )
}

fn write_artifact(path: &Path, artifact: &IntentModelArtifact) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, artifact)
        .context("failed to serialize intent model")?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn read_artifact(path: &Path) -> Result<IntentModelArtifact> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let artifact: IntentModelArtifact = serde_json::from_reader(BufReader::new(file))
        .context("failed to deserialize intent model")?;
    artifact.validate()?;
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn sample(id: &str, session: &str, text: &str, intent: &str) -> ExperimentSample {
        ExperimentSample {
            event_id: id.to_owned(),
            session_id: session.to_owned(),
            transcript: text.to_owned(),
            intent: intent.to_owned(),
        }
    }

    fn toy_artifact() -> IntentModelArtifact {
        let samples = [
            sample("1", "train", "который час", "tell_time"),
            sample("2", "train", "включи музыку", "play_music"),
            sample("3", "train", "таймер на минуту", "set_timer"),
            sample("4", "train", "я читаю книгу", "unknown"),
        ];
        let texts = samples
            .iter()
            .map(|sample| sample.transcript.clone())
            .collect::<Vec<_>>();
        let targets = samples
            .iter()
            .map(|sample| sample.intent.clone())
            .collect::<Vec<_>>();
        let feature_config = FeatureConfig::default();
        let vectorizer = TfidfVectorizer::fit(&texts, feature_config);
        let features = texts
            .iter()
            .map(|text| vectorizer.transform(text))
            .collect::<Vec<_>>();
        let optimization = TrainingConfig {
            epochs: 600,
            reject_threshold: 0.4,
            ..TrainingConfig::default()
        };
        let (
            classifier,
            logistic::TrainingStats {
                initial_loss,
                final_loss,
            },
        ) = LogisticRegression::train(&features, &targets, vectorizer.dimensions(), optimization)
            .unwrap();
        IntentModelArtifact {
            schema_version: MODEL_SCHEMA_VERSION,
            feature_extractor: vectorizer,
            classifier,
            training: TrainingMetadata {
                samples: samples.len(),
                sessions: vec!["train".to_owned()],
                class_counts: class_counts(&samples),
                feature_config,
                optimization,
                initial_loss,
                final_loss,
            },
        }
    }

    fn test_directory() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "jarvis-intent-ml-test-{}-{unique}-{}",
            std::process::id(),
            TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn training_is_deterministic() {
        assert_eq!(toy_artifact(), toy_artifact());
    }

    #[test]
    fn train_command_writes_a_stable_reloadable_artifact() {
        let root = test_directory();
        let splits = root.join("splits");
        fs::create_dir_all(&splits).unwrap();
        let samples = [
            sample("1", "a", "который час", "tell_time"),
            sample("2", "a", "сколько времени", "tell_time"),
            sample("3", "b", "включи музыку", "play_music"),
            sample("4", "b", "открой spotify", "play_music"),
            sample("5", "c", "таймер на минуту", "set_timer"),
            sample("6", "c", "поставь таймер", "set_timer"),
            sample("7", "a", "я читаю книгу", "unknown"),
            sample("8", "b", "не включай музыку", "unknown"),
        ];
        let jsonl = samples
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
            .join("\n");
        fs::write(splits.join("train.jsonl"), format!("{jsonl}\n")).unwrap();
        let first = root.join("first.json");
        let second = root.join("second.json");

        train(&splits, &first).unwrap();
        train(&splits, &second).unwrap();

        assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
        let artifact = read_artifact(&first).unwrap();
        assert_eq!(artifact.training.samples, 8);
        assert_eq!(artifact.training.sessions.len(), 3);
        assert!(artifact.training.final_loss < artifact.training.initial_loss);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn artifact_round_trip_preserves_predictions() {
        let artifact = toy_artifact();
        let json = serde_json::to_string(&artifact).unwrap();
        let decoded: IntentModelArtifact = serde_json::from_str(&json).unwrap();

        decoded.validate().unwrap();
        assert_eq!(
            artifact.predict("включи музыку"),
            decoded.predict("включи музыку")
        );
    }

    #[test]
    fn refuses_session_leakage() {
        let samples = vec![sample("1", "train", "который час", "tell_time")];
        let error = ensure_independent_sessions(&["train".to_owned()], &samples).unwrap_err();
        assert!(error.to_string().contains("session leakage"));
    }

    #[test]
    fn safety_metrics_count_false_commands_on_unknown() {
        let artifact = toy_artifact();
        let samples = vec![
            sample("1", "test", "который час", "tell_time"),
            sample("2", "test", "включи музыку", "unknown"),
        ];

        let metrics = evaluate_strategy(&artifact, &samples, Strategy::Rules);

        assert_eq!(metrics.correct, 1);
        assert_eq!(metrics.supported_correct, 1);
        assert_eq!(metrics.false_commands, 1);
        assert_eq!(metrics.negatives, 1);
    }

    #[test]
    fn hybrid_requires_ml_agreement_before_accepting_a_rule_command() {
        assert_eq!(hybrid_prediction("tell_time", "tell_time"), "tell_time");
        assert_eq!(hybrid_prediction("tell_time", "unknown"), "unknown");
        assert_eq!(hybrid_prediction("tell_time", "play_music"), "unknown");
        assert_eq!(hybrid_prediction("unknown", "play_music"), "play_music");
    }
}
