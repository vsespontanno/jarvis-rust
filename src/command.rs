#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    TellTime,
    PlayMusic,
    Unknown { text: String },
}
