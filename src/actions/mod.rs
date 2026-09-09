mod music;
mod time;
mod timer;

use anyhow::Result;

use crate::command::Command;

pub use timer::{TimerElapsed, TimerScheduler, channel as timer_channel, notify as notify_timer};

pub fn execute(command: &Command, timers: &TimerScheduler) -> Result<Option<String>> {
    match command {
        Command::TellTime => Ok(Some(time::current_time_message())),
        Command::PlayMusic => music::open_spotify().map(Some),
        Command::SetTimer { duration_seconds } => timers.start(*duration_seconds).map(Some),
        Command::NoSpeech | Command::Unknown { .. } => Ok(None),
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
        let (timers, _events) = timer_channel();

        assert_eq!(execute(&command, &timers).unwrap(), None);
    }
}
