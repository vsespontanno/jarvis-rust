# Jarvis

A local voice assistant built incrementally in Rust. The current version records five seconds from
the default microphone, converts the audio to mono 16 kHz PCM, and transcribes Russian speech with
Whisper.

## Prerequisites

On Apple Silicon macOS, install CMake to build the bundled `whisper.cpp` library:

```bash
brew install cmake
```

Download the multilingual Whisper `small` model:

```bash
mkdir -p models
curl --fail --location \
  --output models/ggml-small.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
shasum -a 1 models/ggml-small.bin
```

Expected SHA-1:

```text
55356645c2b361a969dfd0ef2c5a50d530afd8d5
```

Models whose names end in `.en` support English only and cannot transcribe Russian.

## Run

```bash
cargo run --release
```

The default model path is `models/ggml-small.bin`. A different model can be supplied as the first
argument:

```bash
cargo run --release -- /path/to/ggml-model.bin
```

The first launch may require enabling microphone access for Terminal in **System Settings → Privacy
& Security → Microphone**.
