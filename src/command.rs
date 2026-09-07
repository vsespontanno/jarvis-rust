#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    TellTime,
    PlayMusic,
    NoSpeech,
    Unknown { text: String },
}

impl Command {
    pub fn intent_name(&self) -> &'static str {
        match self {
            Self::TellTime => "tell_time",
            Self::PlayMusic => "play_music",
            Self::NoSpeech => "no_speech",
            Self::Unknown { .. } => "unknown",
        }
    }
}
