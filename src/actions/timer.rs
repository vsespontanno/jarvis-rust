use std::{
    io::{self, Write},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender},
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, ensure};

const ALERT_SOUND_PATH: &str = "/System/Library/Sounds/Submarine.aiff";

#[derive(Clone)]
pub struct TimerScheduler {
    sender: Sender<TimerElapsed>,
}

pub struct TimerElapsed {
    duration_seconds: u64,
}

pub fn channel() -> (TimerScheduler, Receiver<TimerElapsed>) {
    let (sender, receiver) = std::sync::mpsc::channel();
    (TimerScheduler { sender }, receiver)
}

impl TimerScheduler {
    pub fn start(&self, duration_seconds: u64) -> Result<String> {
        let sender = self.sender.clone();
        let formatted_duration = format_duration(duration_seconds);

        thread::Builder::new()
            .name("jarvis-timer".to_owned())
            .spawn(move || {
                thread::sleep(Duration::from_secs(duration_seconds));
                let _ = sender.send(TimerElapsed { duration_seconds });
            })
            .context("failed to start the timer thread")?;

        Ok(format!("Таймер установлен на {formatted_duration}."))
    }
}

pub fn notify(timer: TimerElapsed) -> Result<()> {
    let message = format!(
        "Таймер на {} завершён.",
        format_duration(timer.duration_seconds)
    );
    eprintln!("\nJarvis:\n{message}\x07");
    let _ = io::stderr().flush();

    let sound_active = Arc::new(AtomicBool::new(true));
    let sound_flag = Arc::clone(&sound_active);
    let sound_thread = thread::Builder::new()
        .name("jarvis-timer-sound".to_owned())
        .spawn(move || play_sound_until_stopped(&sound_flag))
        .context("failed to start the timer sound thread")?;

    let alert_result = show_alert(&message);
    sound_active.store(false, Ordering::Relaxed);
    let sound_result = sound_thread
        .join()
        .map_err(|_| anyhow!("timer sound thread panicked"))?;

    alert_result?;
    sound_result
}

fn play_sound_until_stopped(active: &AtomicBool) -> Result<()> {
    while active.load(Ordering::Relaxed) {
        let status = Command::new("afplay")
            .arg(ALERT_SOUND_PATH)
            .status()
            .context("failed to start the macOS timer sound")?;
        ensure!(status.success(), "the macOS timer sound failed");
    }
    Ok(())
}

fn show_alert(message: &str) -> Result<()> {
    let script = format!(
        "display alert \"Jarvis\" message \"{message}\" as critical buttons {{\"OK\"}} default button \"OK\""
    );
    let output = Command::new("osascript")
        .args(["-e", &script])
        .output()
        .context("failed to show the macOS timer alert")?;
    ensure!(
        output.status.success(),
        "the macOS timer alert failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

fn format_duration(duration_seconds: u64) -> String {
    let hours = duration_seconds / 3_600;
    let minutes = duration_seconds % 3_600 / 60;
    let seconds = duration_seconds % 60;
    let mut parts = Vec::new();

    if hours > 0 {
        parts.push(format!("{hours} {}", plural(hours, "час", "часа", "часов")));
    }
    if minutes > 0 {
        parts.push(format!(
            "{minutes} {}",
            plural(minutes, "минуту", "минуты", "минут")
        ));
    }
    if seconds > 0 || parts.is_empty() {
        parts.push(format!(
            "{seconds} {}",
            plural(seconds, "секунду", "секунды", "секунд")
        ));
    }

    parts.join(" ")
}

fn plural<'a>(value: u64, one: &'a str, few: &'a str, many: &'a str) -> &'a str {
    let last_two = value % 100;
    let last = value % 10;

    if last == 1 && last_two != 11 {
        one
    } else if (2..=4).contains(&last) && !(12..=14).contains(&last_two) {
        few
    } else {
        many
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn formats_timer_duration_with_russian_plural_forms() {
        assert_eq!(format_duration(1), "1 секунду");
        assert_eq!(format_duration(22), "22 секунды");
        assert_eq!(format_duration(60), "1 минуту");
        assert_eq!(format_duration(300), "5 минут");
        assert_eq!(format_duration(3_900), "1 час 5 минут");
    }

    #[test]
    fn scheduler_reports_elapsed_timer_without_blocking_the_caller() {
        let (timers, events) = channel();

        assert_eq!(timers.start(0).unwrap(), "Таймер установлен на 0 секунд.");
        let event = events.recv_timeout(Duration::from_secs(1)).unwrap();

        assert_eq!(event.duration_seconds, 0);
    }
}
