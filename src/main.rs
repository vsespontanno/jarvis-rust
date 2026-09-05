mod audio;

use std::{path::Path, time::Duration};

use anyhow::{Context, Result};

const RECORDING_DURATION: Duration = Duration::from_secs(5);
const OUTPUT_PATH: &str = "recording.wav";

fn main() -> Result<()> {
    println!("Recording from the default microphone for 5 seconds...");

    let recording = audio::record_default_input(RECORDING_DURATION)
        .context("failed to record audio from the default input device")?;

    recording
        .write_wav(Path::new(OUTPUT_PATH))
        .context("failed to write the recording")?;

    println!(
        "Saved {} samples ({} Hz, {} channel(s)) to {OUTPUT_PATH}",
        recording.samples.len(),
        recording.sample_rate,
        recording.channels,
    );

    Ok(())
}
