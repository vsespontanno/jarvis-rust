use std::{collections::BTreeSet, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
