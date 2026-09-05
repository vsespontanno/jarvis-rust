use std::{
    path::Path,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use cpal::{
    Device, SampleFormat, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

pub struct Recording {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
    pub channels: u16,
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
}

pub fn record_default_input(duration: Duration) -> Result<Recording> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default input device is available")?;
    let supported_config = device
        .default_input_config()
        .context("could not read the default input configuration")?;

    let sample_format = supported_config.sample_format();
    let config: StreamConfig = supported_config.into();
    let samples = Arc::new(Mutex::new(Vec::with_capacity(
        config.sample_rate as usize * config.channels as usize * duration.as_secs() as usize,
    )));
    let device_name = device
        .description()
        .map(|description| description.name().to_owned())
        .unwrap_or_else(|_| "unknown device".into());

    println!(
        "Input: {} — {} Hz, {} channel(s), {sample_format:?}",
        device_name, config.sample_rate, config.channels,
    );

    let stream = build_input_stream(&device, &config, sample_format, Arc::clone(&samples))?;
    stream.play().context("could not start the input stream")?;
    thread::sleep(duration);
    drop(stream);

    let samples = Arc::try_unwrap(samples)
        .map_err(|_| anyhow::anyhow!("audio callback still owns the sample buffer"))?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("audio sample buffer mutex was poisoned"))?;

    Ok(Recording {
        samples,
        sample_rate: config.sample_rate,
        channels: config.channels,
    })
}

fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    samples: Arc<Mutex<Vec<i16>>>,
) -> Result<Stream> {
    let error_callback = |error| eprintln!("Audio stream error: {error}");

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            *config,
            move |data: &[f32], _| append_samples(&samples, data, f32_to_i16),
            error_callback,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            *config,
            move |data: &[i16], _| append_samples(&samples, data, |sample| sample),
            error_callback,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            *config,
            move |data: &[u16], _| append_samples(&samples, data, u16_to_i16),
            error_callback,
            None,
        )?,
        other => bail!("unsupported input sample format: {other:?}"),
    };

    Ok(stream)
}

fn append_samples<T: Copy>(destination: &Mutex<Vec<i16>>, input: &[T], convert: impl Fn(T) -> i16) {
    if let Ok(mut destination) = destination.try_lock() {
        destination.extend(input.iter().copied().map(convert));
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
}
