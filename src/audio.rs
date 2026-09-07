use std::{
    ops::Range,
    path::Path,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel},
    },
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use cpal::{
    Device, SampleFormat, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

pub struct Recording {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
    pub channels: u16,
    pub device_name: String,
}

#[derive(Clone, Copy, Debug)]
pub struct AudioLevel {
    pub rms: f32,
    pub dbfs: f32,
    /// Exclusive native-rate frame index covered by this measurement.
    pub end_frame: usize,
}

pub struct RecordingSession {
    stream: Stream,
    samples: Arc<Mutex<Vec<i16>>>,
    levels: Receiver<AudioLevel>,
    sample_rate: u32,
    channels: u16,
    device_name: String,
}

impl Recording {
    pub fn write_wav(&self, path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;

        for &sample in &self.samples {
            writer.write_sample(sample)?;
        }

        writer.finalize()?;
        Ok(())
    }

    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    pub fn slice_frames(&self, range: Range<usize>) -> Result<Self> {
        ensure!(range.start <= range.end, "invalid audio frame range");
        ensure!(
            range.end <= self.frame_count(),
            "audio frame range exceeds the recording"
        );

        let channels = self.channels as usize;
        let sample_range = range.start * channels..range.end * channels;

        Ok(Self {
            samples: self.samples[sample_range].to_vec(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            device_name: self.device_name.clone(),
        })
    }
}

impl RecordingSession {
    pub fn start(expected_duration: Duration, level_window: Duration) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("no default input device is available")?;
        let supported_config = device
            .default_input_config()
            .context("could not read the default input configuration")?;

        let sample_format = supported_config.sample_format();
        let config: StreamConfig = supported_config.into();
        let expected_frames =
            (config.sample_rate as f64 * expected_duration.as_secs_f64()).ceil() as usize;
        let capacity = expected_frames
            .checked_mul(config.channels as usize)
            .context("recording buffer capacity overflowed")?;
        let samples = Arc::new(Mutex::new(Vec::with_capacity(capacity)));
        let level_window_frames =
            (config.sample_rate as f64 * level_window.as_secs_f64()).round() as usize;
        ensure!(
            level_window_frames > 0,
            "audio level window must contain at least one frame"
        );
        let level_channel_capacity = expected_frames.div_ceil(level_window_frames) + 1;
        let (level_sender, levels) = sync_channel(level_channel_capacity);

        let device_name = device
            .description()
            .map(|description| description.name().to_owned())
            .unwrap_or_else(|_| "unknown device".into());

        println!(
            "Input: {} — {} Hz, {} channel(s), {sample_format:?}",
            device_name, config.sample_rate, config.channels,
        );

        let stream = build_input_stream(
            &device,
            &config,
            sample_format,
            Arc::clone(&samples),
            level_sender,
            level_window_frames,
        )?;
        stream.play().context("could not start the input stream")?;

        Ok(Self {
            stream,
            samples,
            levels,
            sample_rate: config.sample_rate,
            channels: config.channels,
            device_name,
        })
    }

    pub fn recv_level_timeout(
        &self,
        timeout: Duration,
    ) -> std::result::Result<AudioLevel, RecvTimeoutError> {
        self.levels.recv_timeout(timeout)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn finish(self) -> Result<Recording> {
        drop(self.stream);

        let samples = Arc::try_unwrap(self.samples)
            .map_err(|_| anyhow::anyhow!("audio callback still owns the sample buffer"))?
            .into_inner()
            .map_err(|_| anyhow::anyhow!("audio sample buffer mutex was poisoned"))?;

        Ok(Recording {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            device_name: self.device_name,
        })
    }
}

#[derive(Debug)]
struct LevelAccumulator {
    sum_squares: f64,
    samples: usize,
    window_samples: usize,
    total_samples: usize,
}

impl LevelAccumulator {
    fn new(window_samples: usize) -> Self {
        Self {
            sum_squares: 0.0,
            samples: 0,
            window_samples,
            total_samples: 0,
        }
    }

    fn push(&mut self, sample: f32) -> Option<AudioLevel> {
        self.sum_squares += f64::from(sample) * f64::from(sample);
        self.samples += 1;
        self.total_samples += 1;

        if self.samples < self.window_samples {
            return None;
        }

        let rms = (self.sum_squares / self.samples as f64).sqrt() as f32;
        let dbfs = 20.0 * rms.max(1.0e-12).log10();
        self.sum_squares = 0.0;
        self.samples = 0;

        Some(AudioLevel {
            rms,
            dbfs,
            end_frame: self.total_samples,
        })
    }
}

fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    samples: Arc<Mutex<Vec<i16>>>,
    level_sender: SyncSender<AudioLevel>,
    level_window_frames: usize,
) -> Result<Stream> {
    let error_callback = |error| eprintln!("Audio stream error: {error}");
    let channels = config.channels as usize;

    let stream = match sample_format {
        SampleFormat::F32 => {
            let mut meter = LevelAccumulator::new(level_window_frames);
            device.build_input_stream(
                *config,
                move |data: &[f32], _| {
                    process_input(
                        &samples,
                        data,
                        f32_to_i16,
                        channels,
                        &mut meter,
                        &level_sender,
                    )
                },
                error_callback,
                None,
            )?
        }
        SampleFormat::I16 => {
            let mut meter = LevelAccumulator::new(level_window_frames);
            device.build_input_stream(
                *config,
                move |data: &[i16], _| {
                    process_input(
                        &samples,
                        data,
                        |sample| sample,
                        channels,
                        &mut meter,
                        &level_sender,
                    )
                },
                error_callback,
                None,
            )?
        }
        SampleFormat::U16 => {
            let mut meter = LevelAccumulator::new(level_window_frames);
            device.build_input_stream(
                *config,
                move |data: &[u16], _| {
                    process_input(
                        &samples,
                        data,
                        u16_to_i16,
                        channels,
                        &mut meter,
                        &level_sender,
                    )
                },
                error_callback,
                None,
            )?
        }
        other => bail!("unsupported input sample format: {other:?}"),
    };

    Ok(stream)
}

fn process_input<T: Copy>(
    destination: &Mutex<Vec<i16>>,
    input: &[T],
    convert: impl Fn(T) -> i16,
    channels: usize,
    meter: &mut LevelAccumulator,
    level_sender: &SyncSender<AudioLevel>,
) {
    if let Ok(mut destination) = destination.try_lock() {
        for frame in input.chunks_exact(channels) {
            let mut mono = 0.0;

            for &sample in frame {
                let sample = convert(sample);
                destination.push(sample);
                mono += sample as f32 / 32_768.0;
            }

            if let Some(level) = meter.push(mono / channels as f32) {
                let _ = level_sender.try_send(level);
            }
        }
    }
}

fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

fn u16_to_i16(sample: u16) -> i16 {
    (sample as i32 - 32_768) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_float_samples_to_signed_pcm() {
        assert_eq!(f32_to_i16(-1.0), i16::MIN + 1);
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), i16::MAX);
        assert_eq!(f32_to_i16(2.0), i16::MAX);
    }

    #[test]
    fn recenters_unsigned_pcm() {
        assert_eq!(u16_to_i16(0), i16::MIN);
        assert_eq!(u16_to_i16(32_768), 0);
        assert_eq!(u16_to_i16(u16::MAX), i16::MAX);
    }

    #[test]
    fn calculates_rms_and_dbfs_for_complete_window() {
        let mut meter = LevelAccumulator::new(4);

        assert!(meter.push(0.5).is_none());
        assert!(meter.push(-0.5).is_none());
        assert!(meter.push(0.5).is_none());
        let level = meter.push(-0.5).unwrap();

        assert!((level.rms - 0.5).abs() < 1.0e-6);
        assert!((level.dbfs - -6.020_6).abs() < 1.0e-4);
    }

    #[test]
    fn starts_a_new_level_window_after_emitting() {
        let mut meter = LevelAccumulator::new(2);

        assert!(meter.push(1.0).is_none());
        assert!(meter.push(1.0).is_some());
        assert!(meter.push(0.0).is_none());
        let silence = meter.push(0.0).unwrap();

        assert_eq!(silence.rms, 0.0);
        assert_eq!(silence.dbfs, -240.0);
    }
}
