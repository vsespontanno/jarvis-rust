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

Download the default multilingual Whisper `large-v3-turbo` model:

```bash
mkdir -p models
curl --fail --location \
  --output models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
shasum -a 1 models/ggml-large-v3-turbo.bin
```

Expected SHA-1:

```text
4af2b29d7ec73d781377bfd1758ca957a807e941
```

Models whose names end in `.en` support English only and cannot transcribe Russian.

The smaller multilingual `small` model remains available as a faster alternative:

```bash
curl --fail --location \
  --output models/ggml-small.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
shasum -a 1 models/ggml-small.bin
```

Its expected SHA-1 is `55356645c2b361a969dfd0ef2c5a50d530afd8d5`. The local model files are
excluded from Git; `small` is about 465 MB and `large-v3-turbo` is about 1.5 GB.

## Run

```bash
cargo run --release
```

Jarvis returns to listening after each command. Stop it gracefully with `Ctrl+C`.

The default microphone stream is opened once and stays alive until shutdown. VAD performs its
one-second calibration only after startup; later utterances reuse the adaptive noise-floor estimate.
Audio arriving while Whisper or an action is running is intentionally discarded instead of being
queued as a delayed command.

The default model path is `models/ggml-large-v3-turbo.bin`. A different model can be supplied as
the first argument:

```bash
cargo run --release -- /path/to/ggml-model.bin
```

For example, switch back to the faster `small` model using:

```bash
cargo run --release -- models/ggml-small.bin
```

Controlled collection uses the same default model. Historical events retain their Whisper model in
provenance, so sessions collected with `small` remain distinguishable. Offline benchmarking accepts
an explicit model path.

Native Whisper diagnostics are hidden unless they are warnings or errors. Enable detailed logs when
debugging with:

```bash
RUST_LOG=debug cargo run --release
```

The first launch may require enabling microphone access for Terminal in **System Settings → Privacy
& Security → Microphone**.

## CLI reference

Run `cargo run --release -- --help` for the built-in reference. All current invocation forms are:

| Command | Purpose | Default when omitted |
| --- | --- | --- |
| `cargo run --release` | Start the continuous voice-assistant loop | `models/ggml-large-v3-turbo.bin` |
| `cargo run --release -- MODEL_PATH` | Start Jarvis with another Whisper model | — |
| `cargo run --release -- --dataset-report [EVENTS_PATH]` | Report dataset, VAD, Whisper, and parser quality | `data/events.jsonl` |
| `cargo run --release -- --dataset-review [DATASET_ROOT]` | Label all previously unreviewed events | `data` |
| `cargo run --release -- --dataset-relabel EVENT_ID [DATASET_ROOT]` | Replace one label append-only | `data` |
| `cargo run --release -- --dataset-collect [PLAN_PATH]` | Run prompted collection without executing actions | `collection-plans/session-01.json` |
| `cargo run --release -- --dataset-export [OUTPUT_DIR]` | Export deterministic session-based splits | `data/splits` |
| `cargo run --release -- --dataset-stt-benchmark MODEL_PATH [CAMPAIGN]` | Re-transcribe labeled WAV files and compare STT models | all eligible campaigns |
| `cargo run --release -- --intent-train [SPLITS_DIR] [MODEL_PATH]` | Train the offline TF-IDF logistic-regression intent model | `data/splits`, `data/models/intent-v1.json` |
| `cargo run --release -- --intent-evaluate [validation\|test] [SPLITS_DIR] [MODEL_PATH]` | Compare rules, ML, and the safe hybrid | `validation`, `data/splits`, `data/models/intent-v1.json` |
| `cargo run --release -- --help` | Print this reference without loading the microphone or model | — |

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
- `поставь таймер на 30 секунд`
- `поставь таймер на пять минут`
- `включи таймер на 1 час 30 минут`

Music commands open the installed Spotify application using the standard macOS application
launcher. The parser also accepts observed Whisper substitutions such as `открою Spotify` and
`спотик`. Timer parsing accepts the observed code-switch `timer` and the verb forms `поставим` and
`поставив` when a valid duration follows.

If the entire Whisper transcript is formatted as a sound annotation in square brackets,
parentheses, or asterisks—for example `[музыка]`, `(звук от джанра)`, or `*хм*`—Jarvis records it as
the `no_speech` intent and rejects it without executing an action. Wrapped text is always rejected,
even if it happens to contain command words; ordinary unwrapped commands are unaffected.

Timers run in Jarvis itself without blocking the listening loop. When a timer expires, audio capture
is paused, Jarvis repeats the macOS `Submarine` sound while a visible alert waits for the `OK`
button. Pressing `OK` stops the sound and resumes capture, so the alert cannot activate Jarvis
itself. Active timers are not persisted and are cancelled when the Jarvis process exits. The first
version accepts durations up to 24 hours using digits or Russian number words from one through
ninety-nine.

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
├── events.jsonl
├── labels.jsonl
└── utterances/
    └── <unique-id>.wav
```

Each line in `events.jsonl` is an independent JSON record containing:

- schema version, unique ID, and UTC timestamp;
- relative audio path and input-device metadata;
- Whisper transcript;
- predicted intent and extracted slots;
- action status, response, or error;
- aggregate VAD telemetry;
- legacy `ground_truth` fields when reading schema-version 1 events.

Schema-version 2 events also include the Jarvis version, process session ID, Whisper model, parser
version, and the VAD configuration used for that prediction. Schema-version 3 adds optional
controlled-collection metadata. Actual noise floor and adaptive thresholds remain in per-event VAD
telemetry. Older records remain readable.

Raw events are never rewritten. Human ground truth is appended separately to `labels.jsonl` and
linked to its source event by ID; if an event is labeled again, the latest label wins.

Collect the first controlled session from the tracked Russian plans with:

```bash
cargo run --release -- --dataset-collect
```

Five non-overlapping plans live in `collection-plans/`. Each contains 16 `tell_time`, 16
`play_music`, 16 `set_timer`, and 32 hard-negative `unknown` prompts. Record one plan per independent
session, ideally changing the device, day, distance, background noise, or speech tempo:

```bash
cargo run --release -- --dataset-collect collection-plans/session-02.json
cargo run --release -- --dataset-collect collection-plans/session-03.json
cargo run --release -- --dataset-collect collection-plans/session-04.json
cargo run --release -- --dataset-collect collection-plans/session-05.json
```

Together the plans produce the initial target of 80 time, 80 music, 80 timer, and 160 unknown
samples without repeating an exact normalized prompt across sessions. Enter starts capture, `s`
skips a prompt, and `q` ends the session. The normal microphone, VAD, Whisper, and parser pipeline
is reused, but voice actions are disabled. Each captured event receives `prompted` source metadata
and an automatic label containing the expected transcript and intent. Natural and prompted samples
therefore remain distinguishable.

Collection resumes from prompt IDs that already have both an event and a label for the same
campaign. It also uses a noise-tolerant VAD profile (`+6 dB` start and `+4 dB` end margins) without
changing the normal Jarvis profile. Short detections and complete VAD timeouts retain their WAV in
prompted mode so false negatives remain available for later VAD and Whisper experiments.

Benchmark a Whisper model offline on labeled audio from one campaign:

```bash
cargo run --release -- --dataset-stt-benchmark models/ggml-small.bin intent-ru-session-01
```

The benchmark never rewrites events, labels, or audio. It creates a model-specific JSON Lines file
under `data/benchmarks/` containing the reference, historical and new transcripts, word-error
counts, latency, provenance, and per-sample errors. Its terminal report compares aggregate WER with
the historical transcript, measures downstream intent accuracy, shows session/device/campaign/VAD
breakdowns and lists the worst and Spotify-related cases. Run the same command with another model
path for a like-for-like comparison.

### Whisper benchmark at the 0.6.0 development checkpoint

This is a `0.6.0-dev` measurement, not a released `0.6.0`: `Cargo.toml` and Changie still identify
the current release as `0.5.0`. The table should be extended after the remaining independent
collection sessions are recorded.

The same 80 labeled prompted samples from `intent-ru-session-01` were transcribed on the MacBook
microphone dataset recorded in a noisy cafe. Labels are still provisional, and the campaign
contains background-speech insertions, so this is a practical checkpoint rather than a final model
ranking.

| Metric | `ggml-small` | `ggml-large-v3-turbo` |
| --- | ---: | ---: |
| Exact transcripts | 37/80 | 44/80 |
| Aggregate WER | 29.6% | 36.0% |
| Median sample WER | 20.0% | 0.0% |
| Intent accuracy | 53/80 (66.2%) | 57/80 (71.2%) |
| Supported-command accuracy | 24/48 (50.0%) | 27/48 (56.2%) |
| False commands on negatives | 3/32 (9.4%) | 2/32 (6.2%) |
| Spotify brand retained | 6/13 | 11/13 |
| Spotify aggregate WER | 40.4% | 29.8% |
| Median transcription latency | 160 ms | 598 ms |

On this noisy campaign, `large-v3-turbo` preserved Spotify and downstream intent more reliably but
was about 3.7 times slower and produced more background-speech insertions, which worsened aggregate
WER. It is now the default because command intent and Spotify recognition are more important for
the current interactive checkpoint. Future independent sessions will show whether that choice
should remain permanent.

Reproduce the comparison without recording new audio:

```bash
cargo run --release -- --dataset-stt-benchmark models/ggml-small.bin intent-ru-session-01
cargo run --release -- --dataset-stt-benchmark models/ggml-large-v3-turbo.bin intent-ru-session-01
```

Start or resume interactive review with:

```bash
cargo run --release -- --dataset-review
```

Press Enter to label the current event, `p` to play its WAV on macOS, `s` to skip it, or `q` to
quit. Editable prompts explicitly say that Enter keeps the current value, so only mistakes need
typing. On exit, review reports how many labels were saved, skipped, and remain unfinished. A later
run automatically starts with events that do not yet have a label.

Recheck an already labeled event by ID with:

```bash
cargo run --release -- --dataset-relabel <event-id>
```

The command shows the current label, can replay the WAV with `p`, and uses the existing values as
defaults. Saving appends a replacement line to `labels.jsonl`; it never edits the previous label or
the original event.

The audio uses the microphone's native sample rate and channel count. This preserves more source
information for later experiments than storing only the 16 kHz Whisper input. A failure during
preprocessing, transcription, or action execution is recorded in `processing_error`; it does not
terminate the command loop.

Export training manifests without copying audio files:

```bash
cargo run --release -- --dataset-export data/splits
```

The exporter creates `train.jsonl`, `validation.jsonl`, and `test.jsonl`. Eligible rows contain the
event ID, session ID, raw Whisper transcript, ground-truth intent, `natural`/`prompted` source,
dataset-relative audio path, and assigned split. Non-speech and records missing a label, transcript,
session ID, valid intent, or audio file are skipped and reported.

Sessions are sorted by a stable FNV-1a hash and assigned as complete units. With five sessions this
gives three train, one validation, and one test session; no session can occur in multiple files.
The same dataset produces byte-identical manifests on repeated export. The exporter also reports
per-split class/session counts and warns about missing classes or exact normalized transcript
duplicates across splits. Since every row retains `session_id`, the same manifests can later support
leave-one-session-out evaluation.

## Offline intent ML experiment

The first custom intent model is deliberately implemented without an ML framework. It is not
connected to the live Jarvis runtime. Train it from `train.jsonl` with:

```bash
cargo run --release -- --intent-train
```

The resulting `data/models/intent-v1.json` stores the normalization and n-gram configuration,
train-only vocabulary, IDF values, labels, weights, biases, optimization settings, loss values,
class counts, and training session IDs. This makes every learned value inspectable and allows an
evaluation run to reject session leakage.

Evaluate only on a held-out session split:

```bash
cargo run --release -- --intent-evaluate validation
cargo run --release -- --intent-evaluate test
```

The report compares the existing rule parser, the ML classifier, and an agreement-gated hybrid. If
rules find a supported command, ML must agree before it is accepted; disagreement safely becomes
`unknown`. When rules return `unknown`, a sufficiently confident ML prediction may recover the
command. The three strategies share accuracy, supported-command accuracy, false-command rate,
per-class precision/recall/F1, and confusion-matrix calculations. An empty split or a session
shared with training is an error rather than a publishable metric.

### Character n-grams and TF-IDF

Input is lowercased, punctuation becomes spaces, and repeated whitespace is collapsed. Boundary
markers are added before extracting every Unicode character n-gram of length 2 through 5. If
$c_{j,d}$ is the count of n-gram $j$ in document $d$, its term frequency is

$$
TF_{j,d}=\frac{c_{j,d}}{\sum_k c_{k,d}}.
$$

The vocabulary and document frequencies are fitted on `train.jsonl` only. With $N$ training
documents and $DF_j$ documents containing feature $j$, smoothed inverse document frequency is

$$
IDF_j=\ln\left(\frac{N+1}{DF_j+1}\right)+1.
$$

The unnormalized feature value and its L2-normalized value are

$$
v_{j,d}=TF_{j,d}\,IDF_j,
\qquad
x_{j,d}=\frac{v_{j,d}}{\sqrt{\sum_k v_{k,d}^2}}.
$$

Character features are useful here because Whisper errors often preserve fragments of words such
as `spotify`, `спотифай`, or inflected Russian commands even when complete word matching fails.

### Multinomial logistic regression

For intent class $c$, the model calculates a linear score from the sparse TF-IDF vector:

$$
z_c=\mathbf{w}_c^T\mathbf{x}+b_c.
$$

Scores become class probabilities through numerically stable softmax. Subtracting the largest
score does not change the probabilities but prevents exponent overflow:

$$
p_c=\frac{\exp(z_c-z_{max})}{\sum_k\exp(z_k-z_{max})}.
$$

Training minimizes average multiclass cross-entropy with L2 weight regularization:

$$
J=-\frac{1}{N}\sum_{i=1}^{N}\ln p_{i,y_i}
  +\frac{\lambda}{2}\sum_c\sum_j w_{c,j}^2.
$$

For class $c$ and feature $j$, the full-batch gradients are

$$
\frac{\partial J}{\partial w_{c,j}}
=\frac{1}{N}\sum_{i=1}^{N}
\left(p_{i,c}-\mathbb{1}[y_i=c]\right)x_{i,j}+\lambda w_{c,j},
$$

$$
\frac{\partial J}{\partial b_c}
=\frac{1}{N}\sum_{i=1}^{N}
\left(p_{i,c}-\mathbb{1}[y_i=c]\right).
$$

Our training loop applies ordinary gradient descent:

$$
w_{c,j}\leftarrow w_{c,j}-\eta\frac{\partial J}{\partial w_{c,j}},
\qquad
b_c\leftarrow b_c-\eta\frac{\partial J}{\partial b_c}.
$$

The initial transparent baseline uses 400 epochs, learning rate $\eta=0.5$, regularization
$\lambda=10^{-4}$, and no random initialization or shuffling, so repeated training on identical
input is deterministic. These are baseline values, not tuned final hyperparameters.

### Safe rejection and evaluation metrics

Let $p_{max}=\max_c p_c$. The ML prediction is forced to `unknown` when confidence is below the
stored threshold $\tau=0.60$:

$$
\hat y=
\begin{cases}
\operatorname*{arg\,max}_c p_c, & p_{max}\ge\tau,\\
\texttt{unknown}, & p_{max}<\tau.
\end{cases}
$$

For each class, the evaluation report calculates

$$
Precision=\frac{TP}{TP+FP},
\qquad
Recall=\frac{TP}{TP+FN},
\qquad
F_1=\frac{2\,Precision\,Recall}{Precision+Recall}.
$$

Overall and supported-command accuracy are

$$
Accuracy=\frac{\text{correct predictions}}{\text{all samples}},
\qquad
SupportedAccuracy=
\frac{\text{correct supported-command predictions}}
{\text{all supported-command samples}}.
$$

Safety remains a separate first-class metric:

$$
FalseCommandRate=
\frac{\text{negative samples predicted as a supported command}}
{\text{all negative samples}}.
$$

The threshold must later be selected on validation data, with special attention to false commands.
The test split is reserved for the final frozen comparison and must not be used to tune features,
optimization, or $\tau$.

## MVP direction

The intended MVP pipeline remains:

```text
microphone → continuous listening → VAD → Whisper → custom intent classifier
→ slot extraction → action → spoken response → listening
```

The compact target vocabulary is `tell_time`, `set_timer`, `play_music`, `pause_music`,
`next_track`, volume control, `open_app`, and mandatory `unknown`. The current ML experiment uses
only the four classes already present in labeled data. New actionable classes enter training only
after their actions, slot contracts, positive examples, and hard-negative examples are defined.
After held-out rules/ML/hybrid comparison, the remaining MVP checkpoints are wake-word gating,
simple slot extraction, live classifier integration with confidence rejection, and local TTS. An
LLM planner is explicitly outside the MVP.

Print a read-only summary without loading the microphone or Whisper model:

```bash
cargo run --release -- --dataset-report
```

Pass a different JSON Lines file after the flag when needed:

```bash
cargo run --release -- --dataset-report /path/to/events.jsonl
```

When labels exist beside the event file, the report evaluates each pipeline level separately:

- speech detection: real speech, false activations, early rejections, and non-speech sent to
  Whisper;
- Whisper: normalized word errors on real speech only;
- parser: historical and current accuracy, errors, and intent confusion on supported commands;
- safety: false actionable intents on labeled non-speech and unsupported speech;
- coverage: sessions, devices, natural/prompted intent counts, campaigns, and legacy samples that
  predate session provenance;
- VAD: separate count/min/median/mean/max telemetry for speech and non-speech;
- minimum speech duration: an offline threshold sweep from 40 through 500 ms showing speech recall
  and noise rejection.

Historical predictions remain unchanged. The current parser is replayed against the transcript
saved at capture time, and threshold candidates are recommendations only: the live VAD setting is
never changed by the report. Schema-version 1 events use the historical 20 ms window size when
simulating speech duration.

Before comparing transcripts, the report lowercases text, removes punctuation, and collapses
whitespace. Word Error Rate is then calculated as

$$
WER=\frac{S+D+I}{N},
$$

where $S$ is substituted words, $D$ is deleted words, $I$ is inserted words, and $N$ is the number
of words in the manually corrected transcript. Speech recall and noise rejection for a simulated
threshold are

$$
R_{speech}=\frac{speech\ kept}{speech\ kept+speech\ rejected},
\qquad
R_{noise}=\frac{noise\ rejected}{noise\ kept+noise\ rejected}.
$$

These are offline calculations over labeled detections, not a measurement of all the silence that
never activated Jarvis.

For negative samples, the false-command rate is

$$
FCR=\frac{false\ actionable\ predictions}{negative\ samples}.
$$

A negative sample is either non-speech or speech whose correct intent is `unknown`/`no_speech`.
This guards against improving command recall by making Jarvis dangerously eager to act.

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
