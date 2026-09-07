#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    TellTime,
    Unknown { text: String },
}
