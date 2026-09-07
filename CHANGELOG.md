# Changelog

All notable changes to Jarvis will be documented in this file. The project has not published a
versioned release yet, so current work is collected under `Unreleased`.

## [Unreleased]

### Added

- A continuous command loop that keeps one Whisper model loaded for the lifetime of the process.
- Graceful shutdown with `Ctrl+C` and recovery from errors limited to a single command.
- A local dataset in `data/` with unique native-format WAV files and append-only JSON Lines events.
- Dataset records for timestamps, input-device metadata, transcripts, predicted intents, slots,
  execution results, processing errors, and future manual ground truth.
- Aggregate VAD telemetry for noise floor, start/end thresholds, signal levels, duration, window
  counts, trailing silence, and the speech-end reason.
- Serialization, dataset-storage, and VAD-telemetry tests.

### Changed

- Audio candidates shorter than 140 ms no longer run through Whisper or produce WAV files. Their
  compact VAD metadata is still recorded with a `rejected` status.
- Annotation-only Whisper results such as `[музыка]`, `[шум]`, and `[тишина]` are preserved for
  analysis but classified as `no_speech` and never executed.
- The fixed `recording.wav` and `recording-16khz-mono.wav` outputs were replaced by per-utterance
  dataset storage. Preprocessed 16 kHz audio now remains in memory.
