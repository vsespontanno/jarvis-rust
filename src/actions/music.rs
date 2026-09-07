use std::process::Command;

use anyhow::{Context, Result, bail};

pub fn open_spotify() -> Result<String> {
    let status = Command::new("open")
        .args(["-a", "Spotify"])
        .status()
        .context("failed to run the macOS application launcher")?;

    if !status.success() {
        bail!("could not open Spotify; make sure the application is installed");
    }

    Ok("Открываю Spotify.".to_owned())
}
