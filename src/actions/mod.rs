mod time;

use crate::command::Command;

pub fn execute(command: &Command) -> Option<String> {
    match command {
        Command::TellTime => Some(time::current_time_message()),
        Command::Unknown { .. } => None,
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

        assert_eq!(execute(&command), None);
    }
}
