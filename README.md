# Jarvis

A local voice assistant built incrementally in Rust. The current version continuously listens for
spoken utterances, converts them to mono 16 kHz PCM, transcribes Russian speech with one long-lived
Whisper model, and executes commands for telling the local time and opening Spotify. Every detected
utterance is also stored as part of a local research dataset.

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

Jarvis returns to listening after each command. Stop it gracefully with `Ctrl+C`.

The default microphone stream is opened once and stays alive until shutdown. VAD performs its
one-second calibration only after startup; later utterances reuse the adaptive noise-floor estimate.
Audio arriving while Whisper or an action is running is intentionally discarded instead of being
queued as a delayed command.

The default model path is `models/ggml-small.bin`. A different model can be supplied as the first
argument:

```bash
cargo run --release -- /path/to/ggml-model.bin
```

Native Whisper diagnostics are hidden unless they are warnings or errors. Enable detailed logs when
debugging with:

```bash
RUST_LOG=debug cargo run --release
```

The first launch may require enabling microphone access for Terminal in **System Settings → Privacy
& Security → Microphone**.

## Changelog workflow

Create one fragment for each notable user-facing or architectural change:

```bash
changie new
```

Choose `Added`, `Changed`, `Fixed`, or `Removed` and describe the result rather than the
implementation details. Small refactors do not need a fragment.

At the next completed checkpoint, batch a minor release and regenerate the changelog:

```bash
changie batch minor
changie merge
```

Then update the version in `Cargo.toml` to match and commit the release. Use `changie batch patch`
instead when a checkpoint contains only backward-compatible fixes. Version `1.0.0` is reserved for
the first stable Jarvis release; early checkpoints use `0.x.y` versions.

## Supported commands

Ask for the current local time using phrases such as:

- `сколько времени`
- `который сейчас час`
- `скажи текущее время`
- `включи музыку`
- `запусти музыку`
- `открой Spotify`

Music commands open the installed Spotify application using the standard macOS application
launcher. The parser also accepts observed Whisper substitutions such as `открою Spotify` and
`спотик`.

The command pipeline is intentionally separated into four stages:

```text
Whisper transcript -> parser -> Command -> action
```

Whisper converts audio into text. The rule-based parser maps that text to a typed `Command`, and the
corresponding action performs the work. Unsupported text becomes `Command::Unknown`, so adding new
commands does not require changing the audio or speech-recognition layers.

## Local dataset

Detected utterances are stored locally and excluded from Git:

```text
data/
├── utterances/
│   └── <unique-id>.wav
└── events.jsonl
```

Each line in `events.jsonl` is an independent JSON record containing:

- schema version, unique ID, and UTC timestamp;
- relative audio path and input-device metadata;
- Whisper transcript;
- predicted intent and extracted slots;
- action status, response, or error;
- aggregate VAD telemetry;
- reserved `ground_truth` fields for a corrected transcript, correct intent, whether speech was
  actually present, and free-form notes.

The audio uses the microphone's native sample rate and channel count. This preserves more source
information for later experiments than storing only the 16 kHz Whisper input. A failure during
preprocessing, transcription, or action execution is recorded in `processing_error`; it does not
terminate the command loop.

Print a read-only summary without loading the microphone or Whisper model:

```bash
cargo run --release -- --dataset-report
```

Pass a different JSON Lines file after the flag when needed:

```bash
cargo run --release -- --dataset-report /path/to/events.jsonl
```

The report includes execution and intent counts, rejected short candidates, Whisper non-speech
annotations, input devices, VAD end reasons, and min/median/p95/max distributions for duration,
speech-window count, and noise floor.

For $n$ sorted observations, the report uses the nearest-rank definition of the 95th percentile:

$$
P_{95}=x_{\lceil 0.95n \rceil}.
$$

Unlike the maximum, p95 shows the upper edge of typical observations without being dominated by a
single extreme sample.

## Math used in the current pipeline

### PCM conversion and normalization

The microphone can provide samples in several PCM formats. Floating-point samples are clamped to
the full-scale interval and converted to signed 16-bit PCM for storage:

$$
s_{i16} = \left\lfloor \max(-1,\min(1,x))\,(2^{15}-1) \right\rceil.
$$

Unsigned 16-bit PCM is recentered around zero:

$$
s_{i16} = s_{u16} - 2^{15}.
$$

Before signal processing, integer PCM is normalized back to floating point:

$$
x = \frac{s_{i16}}{2^{15}}.
$$

This gives approximately $x \in [-1,1)$, independent of the integer representation used by the
audio device.

### Downmix to mono

For a frame with $C$ channels, the mono sample is the arithmetic mean of its channel samples:

$$
x_{mono}[n] = \frac{1}{C}\sum_{c=1}^{C}x_c[n].
$$

Whisper expects one channel, and averaging avoids multiplying the amplitude when several channels
contain the same signal.

### Analysis window size

The level meter analyzes windows of duration $T=20\text{ ms}$. At sample rate $f_s$, a window
contains

$$
N = \left\lfloor f_s T \right\rceil
$$

audio frames. For example, this is 480 frames at 24 kHz and 960 frames at 48 kHz. Using time-based
windows keeps the detector behavior consistent across different microphones.

### RMS signal level

For every analysis window, the root mean square amplitude is

$$
x_{RMS} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1}x[n]^2}.
$$

RMS measures signal energy more usefully than an instantaneous peak: alternating positive and
negative waveform samples do not cancel because they are squared first.

### dBFS

The RMS value is converted to decibels relative to digital full scale:

$$
L_{dBFS} = 20\log_{10}\left(\max(x_{RMS}, \varepsilon)\right),
\qquad \varepsilon=10^{-12}.
$$

An RMS amplitude of 1 corresponds to 0 dBFS, 0.5 is approximately -6.02 dBFS, and quieter signals
have increasingly negative values. The small $\varepsilon$ prevents $\log(0)$ for digital silence.

The terminal bar maps the displayed range $[L_{min},0]$, currently $L_{min}=-60$ dBFS, to
$[0,1]$:

$$
p = \max\left(0,\min\left(1,\frac{L_{dBFS}-L_{min}}{0-L_{min}}\right)\right).
$$

### Resampling to 16 kHz

For input rate $f_{in}$ and Whisper's required output rate $f_{out}=16000$, the resampling ratio is

$$
r = \frac{f_{out}}{f_{in}},
\qquad
N_{out} \approx \left\lceil N_{in}r \right\rceil.
$$

For the 24 kHz AirPods example, $r=2/3$: 116640 input frames become 77760 output frames.

By the Nyquist theorem, a signal sampled at $f_s$ can represent frequencies only below

$$
f_{Nyquist}=\frac{f_s}{2}.
$$

After conversion to 16 kHz, the new Nyquist frequency is 8 kHz. Frequencies above it must be removed
before downsampling or they fold into the speech band as aliasing. The project delegates this
anti-alias filtering and fixed-ratio FFT resampling to `rubato`.

### Adaptive voice activity detection

During the first second, the detector collects 50 level windows, discards digitally gated silence
below -120 dBFS, and uses the median of the remaining values as the initial noise floor:

$$
L_{noise}=\mathrm{median}(L_1,L_2,\ldots,L_{50}).
$$

This filtering matters for devices such as AirPods, which may output exact zero-filled windows when
their own noise gate closes. Without it, the estimated floor can incorrectly become -240 dBFS. The
median is less sensitive than the mean to a few unusually loud calibration windows. While the
detector is waiting for speech, non-gated quiet observations slowly update the estimate with an
exponential moving average:

$$
L_{noise,t}=(1-\alpha)L_{noise,t-1}+\alpha L_t,
\qquad \alpha=0.02.
$$

Speech start and stop use different adaptive thresholds, together with absolute guards against
low-level background sounds:

$$
L_{start}=\max(L_{noise}+12\text{ dB},-35\text{ dBFS}),
\qquad
L_{end}=\max(L_{noise}+6\text{ dB},-40\text{ dBFS}).
$$

This difference is hysteresis: once speech has started, the signal may become quieter without
immediately switching back to silence. Speech starts after three consecutive loud windows (60 ms)
and ends after 30 consecutive quiet windows (600 ms). A 300 ms pre-roll is retained before the
detected start so that threshold confirmation does not cut off the first phoneme.

### VAD telemetry

For the $K$ windows classified as speech, Jarvis stores their arithmetic mean and median dBFS:

$$
\overline{L}_{speech}=\frac{1}{K}\sum_{k=1}^{K}L_k,
\qquad
\widetilde{L}_{speech}=\mathrm{median}(L_1,\ldots,L_K).
$$

It also stores the peak window level, the calibrated noise floor, both thresholds, counts of
speech/silence windows, and the stop reason. Saved-audio duration is calculated from its native
frame count:

$$
t_{audio}=1000\frac{N_{frames}}{f_s}\text{ ms}.
$$

For analysis windows of duration $T=20\text{ ms}$, the final silence duration is

$$
t_{silence}=N_{trailing\ quiet}\,T.
$$

These aggregates are small enough to log for every utterance while retaining the information
needed to compare noise conditions, threshold margins, premature stops, and false activations.

Before running Whisper, a detected candidate must contain at least seven speech-classified windows:

$$
t_{minimum}=7\cdot20\text{ ms}=140\text{ ms}.
$$

Shorter candidates are not written to WAV, transcribed, or executed. A compact JSONL event is still
stored with `audio_path: null`, `transcript: null`, VAD telemetry, and the execution status
`rejected`. This keeps repeated clicks and impacts from filling audio storage while retaining their
counts and level statistics. The terminal also reports their measured speech duration for live
diagnostics.

Whisper can describe short non-speech audio with annotations such as `[музыка]`, `[шум]`, or
`[тишина]`. An annotation-only transcript is preserved in the dataset with the predicted intent
`no_speech`, but receives the same `rejected` execution status and never reaches an action.
