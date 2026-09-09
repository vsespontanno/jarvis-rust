# Changelog

All notable changes to Jarvis are documented in this file. The project follows Semantic Versioning
and uses Changie to collect unreleased change fragments.


## 0.3.0 - 2026-09-10
### Added
- Dataset reports can replay saved transcripts through the current parser and summarize changed intent predictions.
- Voice commands can start non-blocking local timers with durations recorded as dataset slots.
### Changed
- The microphone stream stays open for the process lifetime and VAD calibration is reused across consecutive commands.
### Fixed
- Input device names are normalized so incidental surrounding whitespace does not split dataset statistics.
- Fully wrapped Whisper sound annotations are rejected as non-speech instead of reaching command execution.

## 0.2.0 - 2026-09-10
### Added
- A read-only dataset report summarizes execution outcomes, intents, and VAD distributions without loading the speech pipeline.
### Fixed
- Spotify commands recognize additional wording observed in real Whisper transcripts.

## 0.1.0 - 2026-09-08
### Added
- A local Russian voice pipeline with microphone capture, adaptive VAD, audio preprocessing, and Whisper transcription.
- Voice commands for reporting the local time and opening Spotify on macOS.
- Continuous command processing with one long-lived Whisper model and graceful Ctrl+C shutdown.
- Per-utterance JSON Lines dataset logging with native audio, execution outcomes, future ground-truth fields, and aggregate VAD telemetry.
- Short-impulse rejection and safe handling of Whisper non-speech annotations.
