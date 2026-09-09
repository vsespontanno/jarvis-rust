use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    TellTime,
    PlayMusic,
    SetTimer { duration_seconds: u64 },
    NoSpeech,
    Unknown { text: String },
}

impl Command {
    pub fn intent_name(&self) -> &'static str {
        match self {
            Self::TellTime => "tell_time",
            Self::PlayMusic => "play_music",
            Self::SetTimer { .. } => "set_timer",
            Self::NoSpeech => "no_speech",
            Self::Unknown { .. } => "unknown",
        }
    }

    pub fn slots(&self) -> BTreeMap<String, String> {
        match self {
            Self::SetTimer { duration_seconds } => {
                BTreeMap::from([("duration_seconds".to_owned(), duration_seconds.to_string())])
            }
            _ => BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_timer_duration_as_a_dataset_slot() {
        assert_eq!(
            Command::SetTimer {
                duration_seconds: 300
            }
            .slots(),
            BTreeMap::from([("duration_seconds".to_owned(), "300".to_owned())])
        );
        assert!(Command::TellTime.slots().is_empty());
    }
}
