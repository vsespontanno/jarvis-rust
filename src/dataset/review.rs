use std::{
    fs::File,
    io::{self, BufRead, BufReader, Write},
    path::Path,
    process::Command,
};

use anyhow::{Context, Result, bail};

use super::{DatasetLabel, DatasetRecord, LabelStore};
use crate::parser;

pub fn review(root: &Path) -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    review_with(root, stdin.lock(), &mut stdout, play_audio)
}

pub fn relabel(root: &Path, event_id: &str) -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    relabel_with(root, event_id, stdin.lock(), &mut stdout, play_audio)
}

fn review_with(
    root: &Path,
    input: impl BufRead,
    mut output: impl Write,
    mut play: impl FnMut(&Path) -> Result<()>,
) -> Result<()> {
    let events = read_events(&root.join("events.jsonl"))?;
    let labels = LabelStore::open(root)?;
    let labeled = labels.latest()?;
    let pending = events
        .iter()
        .filter(|event| !labeled.contains_key(&event.id))
        .collect::<Vec<_>>();

    if pending.is_empty() {
        writeln!(output, "No unlabeled events.")?;
        return Ok(());
    }

    let mut lines = input.lines();
    let mut saved = 0;
    let mut skipped = 0;
    'events: for (index, event) in pending.iter().enumerate() {
        show_event(&mut output, index + 1, pending.len(), event)?;

        loop {
            let Some(action) = prompt(
                &mut output,
                &mut lines,
                "Action [Enter=label, p=play, s=skip, q=quit]: ",
            )?
            else {
                return write_review_summary(&mut output, saved, skipped, pending.len() - saved);
            };
            match action.to_lowercase().as_str() {
                "" => break,
                "p" => match &event.audio_path {
                    Some(relative_path) => {
                        if let Err(error) = play(&root.join(relative_path)) {
                            writeln!(output, "Could not play audio: {error:#}")?;
                        }
                    }
                    None => writeln!(output, "Audio was not saved for this event.")?,
                },
                "s" => {
                    skipped += 1;
                    continue 'events;
                }
                "q" => {
                    return write_review_summary(
                        &mut output,
                        saved,
                        skipped,
                        pending.len() - saved,
                    );
                }
                _ => writeln!(output, "Use Enter, p, s, or q.")?,
            }
        }

        let Some(label) = prompt_label(&mut output, &mut lines, event, None)? else {
            return write_review_summary(&mut output, saved, skipped, pending.len() - saved);
        };
        labels.append(&label)?;
        saved += 1;
        writeln!(output, "Label saved.\n")?;
    }

    write_review_summary(&mut output, saved, skipped, pending.len() - saved)
}

fn relabel_with(
    root: &Path,
    event_id: &str,
    input: impl BufRead,
    mut output: impl Write,
    mut play: impl FnMut(&Path) -> Result<()>,
) -> Result<()> {
    let events = read_events(&root.join("events.jsonl"))?;
    let event = events
        .iter()
        .find(|event| event.id == event_id)
        .with_context(|| format!("dataset event '{event_id}' was not found"))?;
    let labels = LabelStore::open(root)?;
    let latest = labels.latest()?;
    let current = latest
        .get(event_id)
        .with_context(|| format!("dataset event '{event_id}' has no existing label"))?;

    show_event(&mut output, 1, 1, event)?;
    show_label(&mut output, current)?;
    let mut lines = input.lines();
    loop {
        let Some(action) = prompt(
            &mut output,
            &mut lines,
            "Action [Enter=edit label, p=play, q=quit]: ",
        )?
        else {
            return Ok(());
        };
        match action.to_lowercase().as_str() {
            "" => break,
            "p" => match &event.audio_path {
                Some(relative_path) => {
                    if let Err(error) = play(&root.join(relative_path)) {
                        writeln!(output, "Could not play audio: {error:#}")?;
                    }
                }
                None => writeln!(output, "Audio was not saved for this event.")?,
            },
            "q" => return Ok(()),
            _ => writeln!(output, "Use Enter, p, or q.")?,
        }
    }

    if let Some(label) = prompt_label(&mut output, &mut lines, event, Some(current))? {
        labels.append(&label)?;
        writeln!(output, "Replacement label appended.")?;
    }
    Ok(())
}

fn prompt_label(
    output: &mut impl Write,
    lines: &mut impl Iterator<Item = io::Result<String>>,
    event: &DatasetRecord,
    current: Option<&DatasetLabel>,
) -> Result<Option<DatasetLabel>> {
    let predicted_speech = event.transcript.is_some()
        && event
            .prediction
            .as_ref()
            .is_some_and(|prediction| prediction.intent != "no_speech");
    let default_speech = current.map_or(predicted_speech, |label| label.actual_speech);
    let Some(actual_speech) = prompt_bool(output, lines, "Speech?", default_speech)? else {
        return Ok(None);
    };

    let (corrected_transcript, correct_intent) = if actual_speech {
        let transcript = current
            .and_then(|label| label.corrected_transcript.as_deref())
            .or(event.transcript.as_deref())
            .unwrap_or("");
        let Some(corrected) = prompt_default(output, lines, "Corrected transcript", transcript)?
        else {
            return Ok(None);
        };
        let predicted_intent = current.map_or_else(
            || {
                event
                    .prediction
                    .as_ref()
                    .map_or("unknown", |prediction| prediction.intent.as_str())
            },
            |label| label.correct_intent.as_str(),
        );
        let Some(intent) = prompt_default(output, lines, "Correct intent", predicted_intent)?
        else {
            return Ok(None);
        };
        (Some(corrected), intent)
    } else {
        (None, "no_speech".to_owned())
    };

    let default_notes = current
        .and_then(|label| label.notes.as_deref())
        .unwrap_or("");
    let Some(notes) = prompt_default(output, lines, "Notes", default_notes)? else {
        return Ok(None);
    };
    Ok(Some(DatasetLabel::new(
        event.id.clone(),
        actual_speech,
        corrected_transcript,
        correct_intent,
        (!notes.is_empty()).then_some(notes),
    )))
}

fn show_label(mut output: impl Write, label: &DatasetLabel) -> Result<()> {
    writeln!(output, "Current label:")?;
    writeln!(output, "  Speech: {}", label.actual_speech)?;
    writeln!(
        output,
        "  Transcript: {}",
        label.corrected_transcript.as_deref().unwrap_or("<none>")
    )?;
    writeln!(output, "  Intent: {}", label.correct_intent)?;
    writeln!(
        output,
        "  Notes: {}\n",
        label.notes.as_deref().unwrap_or("<none>")
    )?;
    Ok(())
}

fn write_review_summary(
    mut output: impl Write,
    saved: usize,
    skipped: usize,
    remaining: usize,
) -> Result<()> {
    writeln!(output, "Review summary:")?;
    writeln!(output, "  Labels saved this run: {saved}")?;
    writeln!(output, "  Skipped this run: {skipped}")?;
    writeln!(output, "  Remaining unlabeled: {remaining}")?;
    Ok(())
}

fn read_events(path: &Path) -> Result<Vec<DatasetRecord>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open dataset events at {}", path.display()))?;
    let mut events = Vec::new();

    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.with_context(|| format!("failed to read event line {line_number}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str(&line)
            .with_context(|| format!("invalid event on JSONL line {line_number}"))?;
        events.push(event);
    }

    Ok(events)
}

fn show_event(
    mut output: impl Write,
    index: usize,
    total: usize,
    event: &DatasetRecord,
) -> Result<()> {
    let transcript = event.transcript.as_deref().unwrap_or("<no transcript>");
    let predicted = event
        .prediction
        .as_ref()
        .map_or("<none>", |prediction| prediction.intent.as_str());
    let current = event
        .transcript
        .as_deref()
        .map(|transcript| parser::parse(transcript).intent_name());

    writeln!(output, "[{index}/{total}]\n")?;
    writeln!(output, "ID:\n{}\n", event.id)?;
    writeln!(output, "Transcript:\n\"{transcript}\"\n")?;
    writeln!(output, "Predicted intent:\n{predicted}")?;
    if current.is_some_and(|current| current != predicted) {
        writeln!(output, "Current parser:\n{}", current.unwrap())?;
    }
    writeln!(
        output,
        "\nExecution:\n{}\n",
        event.execution_result.status.as_str()
    )?;
    Ok(())
}

fn prompt(
    output: &mut impl Write,
    lines: &mut impl Iterator<Item = io::Result<String>>,
    message: &str,
) -> Result<Option<String>> {
    write!(output, "{message}")?;
    output.flush()?;
    lines
        .next()
        .transpose()
        .context("failed to read review input")
        .map(|line| line.map(|line| line.trim().to_owned()))
}

fn prompt_bool(
    output: &mut impl Write,
    lines: &mut impl Iterator<Item = io::Result<String>>,
    message: &str,
    default: bool,
) -> Result<Option<bool>> {
    let choices = if default { "Y/n" } else { "y/N" };
    loop {
        let Some(answer) = prompt(output, lines, &format!("{message} [{choices}]: "))? else {
            return Ok(None);
        };
        match answer.to_lowercase().as_str() {
            "" => return Ok(Some(default)),
            "y" | "yes" => return Ok(Some(true)),
            "n" | "no" => return Ok(Some(false)),
            _ => writeln!(output, "Enter y or n.")?,
        }
    }
}

fn prompt_default(
    output: &mut impl Write,
    lines: &mut impl Iterator<Item = io::Result<String>>,
    message: &str,
    default: &str,
) -> Result<Option<String>> {
    prompt(output, lines, &format!("{message} [Enter=keep current]: ")).map(|answer| {
        answer.map(|answer| {
            if answer.is_empty() {
                default.to_owned()
            } else {
                answer
            }
        })
    })
}

fn play_audio(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("audio file does not exist: {}", path.display());
    }
    let status = Command::new("afplay")
        .arg(path)
        .status()
        .context("failed to start the macOS audio player")?;
    if !status.success() {
        bail!("audio playback failed for {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        io::Cursor,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::dataset::{
        ExecutionResult, ExecutionStatus, GroundTruth, InputMetadata, IntentPrediction,
    };

    fn test_directory() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "jarvis-review-test-{}-{unique}",
            std::process::id()
        ))
    }

    fn event(id: &str, transcript: &str) -> DatasetRecord {
        DatasetRecord {
            schema_version: 1,
            id: id.to_owned(),
            timestamp: "2026-09-10T00:00:00Z".to_owned(),
            provenance: None,
            collection: None,
            audio_path: Some(format!("utterances/{id}.wav")),
            input: InputMetadata {
                device: "Test microphone".to_owned(),
                sample_rate_hz: 16_000,
                channels: 1,
            },
            transcript: Some(transcript.to_owned()),
            prediction: Some(IntentPrediction {
                intent: "tell_time".to_owned(),
                slots: BTreeMap::new(),
            }),
            execution_result: ExecutionResult {
                status: ExecutionStatus::Succeeded,
                response: None,
                error: None,
            },
            processing_error: None,
            vad: None,
            ground_truth: GroundTruth::default(),
        }
    }

    fn write_events(root: &Path, events: &[DatasetRecord]) {
        fs::create_dir_all(root).unwrap();
        let jsonl = events
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
            .join("\n");
        fs::write(root.join("events.jsonl"), format!("{jsonl}\n")).unwrap();
    }

    #[test]
    fn accepts_defaults_and_resumes_with_the_next_unlabeled_event() {
        let root = test_directory();
        write_events(
            &root,
            &[
                event("first", "который час"),
                event("second", "сколько времени"),
            ],
        );
        let mut first_output = Vec::new();
        review_with(
            &root,
            Cursor::new("\n\n\n\n\nq\n"),
            &mut first_output,
            |_| Ok(()),
        )
        .unwrap();

        let labels = LabelStore::open(&root).unwrap().latest().unwrap();
        assert_eq!(labels.len(), 1);
        assert_eq!(
            labels["first"].corrected_transcript.as_deref(),
            Some("который час")
        );
        assert_eq!(labels["first"].correct_intent, "tell_time");

        let mut second_output = Vec::new();
        review_with(&root, Cursor::new("q\n"), &mut second_output, |_| Ok(())).unwrap();
        let output = String::from_utf8(second_output).unwrap();
        assert!(output.contains("[1/1]"));
        assert!(output.contains("second"));
        assert!(!output.contains("first"));
        assert!(output.contains("Labels saved this run: 0"));
        assert!(output.contains("Remaining unlabeled: 1"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn plays_and_skips_without_creating_a_label() {
        let root = test_directory();
        write_events(
            &root,
            &[event("first", "который час"), event("second", "таймер")],
        );
        let mut output = Vec::new();
        let mut played = Vec::new();

        review_with(&root, Cursor::new("p\ns\nq\n"), &mut output, |path| {
            played.push(path.to_path_buf());
            Ok(())
        })
        .unwrap();

        assert_eq!(played, vec![root.join("utterances/first.wav")]);
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("Skipped this run: 1"));
        assert!(output.contains("Remaining unlabeled: 2"));
        assert!(
            LabelStore::open(&root)
                .unwrap()
                .latest()
                .unwrap()
                .is_empty()
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accepts_non_speech_as_the_default_for_no_speech_predictions() {
        let root = test_directory();
        let mut noise = event("noise", "[музыка]");
        noise.prediction.as_mut().unwrap().intent = "no_speech".to_owned();
        write_events(&root, &[noise]);
        let mut output = Vec::new();

        review_with(&root, Cursor::new("\n\n\n"), &mut output, |_| Ok(())).unwrap();

        let labels = LabelStore::open(&root).unwrap().latest().unwrap();
        assert!(!labels["noise"].actual_speech);
        assert_eq!(labels["noise"].corrected_transcript, None);
        assert_eq!(labels["noise"].correct_intent, "no_speech");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn appends_a_replacement_for_an_existing_label() {
        let root = test_directory();
        write_events(&root, &[event("spotify", "Открою Spotify")]);
        let labels = LabelStore::open(&root).unwrap();
        labels
            .append(&DatasetLabel::new(
                "spotify".to_owned(),
                true,
                Some("Открою Spotify".to_owned()),
                "unknown".to_owned(),
                None,
            ))
            .unwrap();
        let mut output = Vec::new();
        let mut played = Vec::new();

        relabel_with(
            &root,
            "spotify",
            Cursor::new("p\n\n\n\nplay_music\nchecked by ear\n"),
            &mut output,
            |path| {
                played.push(path.to_path_buf());
                Ok(())
            },
        )
        .unwrap();

        let latest = labels.latest().unwrap();
        assert_eq!(latest["spotify"].correct_intent, "play_music");
        assert_eq!(latest["spotify"].notes.as_deref(), Some("checked by ear"));
        assert_eq!(played, vec![root.join("utterances/spotify.wav")]);
        assert_eq!(
            fs::read_to_string(root.join("labels.jsonl"))
                .unwrap()
                .lines()
                .count(),
            2
        );

        fs::remove_dir_all(root).unwrap();
    }
}
