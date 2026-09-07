mod music;
mod time;

use anyhow::Result;

use crate::command::Command;

pub fn execute(command: &Command) -> Result<Option<String>> {
    match command {
        Command::TellTime => Ok(Some(time::current_time_message())),
        Command::PlayMusic => music::open_spotify().map(Some),
        Command::Unknown { .. } => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_command_has_no_action() {
        let command = Command::Unknown {
            text: "неизвестная команда".to_owned(),
        };

        assert_eq!(execute(&command).unwrap(), None);
    }
}
