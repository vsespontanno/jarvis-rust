use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use super::{DatasetRecord, LabelStore};

const COLLECTION_PLAN_SCHEMA_VERSION: u8 = 1;

#[derive(Debug, Deserialize, PartialEq)]
pub struct CollectionPlan {
    pub schema_version: u8,
    pub campaign: String,
    pub prompts: Vec<CollectionPrompt>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct CollectionPrompt {
    pub id: String,
    pub transcript: String,
    pub intent: String,
}

impl CollectionPlan {
    pub fn load(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read collection plan at {}", path.display()))?;
        let plan: Self = serde_json::from_str(&contents)
            .with_context(|| format!("invalid collection plan at {}", path.display()))?;
        plan.validate()?;
        Ok(plan)
    }

    fn validate(&self) -> Result<()> {
        if self.schema_version != COLLECTION_PLAN_SCHEMA_VERSION {
            bail!(
                "unsupported collection plan schema {}; expected {}",
                self.schema_version,
                COLLECTION_PLAN_SCHEMA_VERSION
            );
        }
        if self.campaign.trim().is_empty() {
            bail!("collection campaign must not be empty");
        }
        if self.prompts.is_empty() {
            bail!("collection plan must contain at least one prompt");
        }

        let supported_intents = ["tell_time", "play_music", "set_timer", "unknown"];
        let mut ids = BTreeSet::new();
        for prompt in &self.prompts {
            if prompt.id.trim().is_empty() || prompt.transcript.trim().is_empty() {
                bail!("collection prompt ID and transcript must not be empty");
            }
            if !supported_intents.contains(&prompt.intent.as_str()) {
                bail!("unsupported expected intent '{}'", prompt.intent);
            }
            if !ids.insert(&prompt.id) {
                bail!("duplicate collection prompt ID '{}'", prompt.id);
            }
        }
        Ok(())
    }
}

pub fn completed_prompt_ids(root: &Path, campaign: &str) -> Result<BTreeSet<String>> {
    let labels = LabelStore::open(root)?.latest()?;
    let events_path = root.join("events.jsonl");
    let file = match fs::File::open(&events_path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to open {}", events_path.display()));
        }
    };

    let mut completed = BTreeSet::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.with_context(|| {
            format!(
                "failed to read {} line {line_number}",
                events_path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let event: DatasetRecord = serde_json::from_str(&line).with_context(|| {
            format!(
                "invalid event on {} line {line_number}",
                events_path.display()
            )
        })?;
        let Some(collection) = event.collection else {
            continue;
        };
        if collection.campaign == campaign && labels.contains_key(&event.id) {
            completed.insert(collection.prompt_id);
        }
    }
    Ok(completed)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::dataset::analysis::normalize_transcript;

    #[test]
    fn validates_a_collection_plan() {
        let plan: CollectionPlan = serde_json::from_str(
            r#"{
                "schema_version": 1,
                "campaign": "baseline-ru-v1",
                "prompts": [
                    {"id":"time-1", "transcript":"который час", "intent":"tell_time"}
                ]
            }"#,
        )
        .unwrap();

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn rejects_duplicate_prompt_ids() {
        let plan = CollectionPlan {
            schema_version: 1,
            campaign: "test".to_owned(),
            prompts: vec![
                CollectionPrompt {
                    id: "same".to_owned(),
                    transcript: "который час".to_owned(),
                    intent: "tell_time".to_owned(),
                },
                CollectionPrompt {
                    id: "same".to_owned(),
                    transcript: "включи музыку".to_owned(),
                    intent: "play_music".to_owned(),
                },
            ],
        };

        assert!(
            plan.validate()
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
    }

    #[test]
    fn bundled_session_plans_reach_targets_without_exact_text_duplicates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("collection-plans");
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut transcripts = [
            "который час",
            "сколько сейчас времени",
            "скажи текущее время",
            "подскажи который сейчас час",
            "включи музыку",
            "открой spotify",
            "запусти spotify",
            "включи пожалуйста музыку",
            "поставь таймер на пять секунд",
            "запусти таймер на одну минуту",
            "поставь таймер на две минуты",
            "поставь на тридцать секунд",
            "как у тебя дела",
            "какая сегодня погода",
            "расскажи что нибудь интересное",
            "я сейчас работаю над проектом",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();

        for session in 1..=5 {
            let path = root.join(format!("session-{session:02}.json"));
            let plan = CollectionPlan::load(&path).unwrap();
            assert_eq!(plan.prompts.len(), 80, "plan: {}", path.display());
            for prompt in plan.prompts {
                *counts.entry(prompt.intent).or_default() += 1;
                assert!(
                    transcripts.insert(normalize_transcript(&prompt.transcript)),
                    "duplicate transcript: {}",
                    prompt.transcript
                );
            }
        }

        assert_eq!(counts["tell_time"], 80);
        assert_eq!(counts["play_music"], 80);
        assert_eq!(counts["set_timer"], 80);
        assert_eq!(counts["unknown"], 160);
    }

    #[test]
    fn resumes_only_prompts_with_an_appended_label() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "jarvis-collection-resume-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let event = |id: &str, prompt_id: &str| {
            serde_json::json!({
                "schema_version": 3,
                "id": id,
                "timestamp": "2026-09-11T00:00:00Z",
                "collection": {
                    "source": "prompted",
                    "campaign": "cafe",
                    "prompt_id": prompt_id,
                    "expected_transcript": "который час",
                    "expected_intent": "tell_time"
                },
                "audio_path": null,
                "input": {"device": "test", "sample_rate_hz": 16000, "channels": 1},
                "transcript": null,
                "prediction": null,
                "execution_result": {"status": "rejected", "response": null, "error": null},
                "processing_error": null,
                "vad": null
            })
            .to_string()
        };
        fs::write(
            root.join("events.jsonl"),
            format!(
                "{}\n{}\n",
                event("labeled", "time-1"),
                event("orphan", "time-2")
            ),
        )
        .unwrap();
        LabelStore::open(&root)
            .unwrap()
            .append(&super::super::DatasetLabel::new(
                "labeled".to_owned(),
                true,
                Some("который час".to_owned()),
                "tell_time".to_owned(),
                None,
            ))
            .unwrap();

        let completed = completed_prompt_ids(&root, "cafe").unwrap();

        assert_eq!(completed, BTreeSet::from(["time-1".to_owned()]));
        fs::remove_dir_all(root).unwrap();
    }
}
