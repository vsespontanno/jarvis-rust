use std::path::Path;

use anyhow::{Result, ensure};
use rubato::{Fft, FixedSync, Resampler, audioadapter_buffers::direct::InterleavedSlice};

use crate::audio::Recording;

pub const STT_SAMPLE_RATE: u32 = 16_000;
const RESAMPLER_CHUNK_SIZE: usize = 1_024;

/// Audio satisfying Whisper's input contract: mono f32 PCM at 16 kHz.
pub struct SttAudio {
    pub samples: Vec<f32>,
}

impl SttAudio {
    pub fn write_wav(&self, path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: STT_SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;

        for &sample in &self.samples {
            writer.write_sample(sample)?;
        }

        writer.finalize()?;
        Ok(())
    }
}

pub fn prepare_for_stt(recording: &Recording) -> Result<SttAudio> {
    ensure!(recording.channels > 0, "channel count must be positive");
    ensure!(recording.sample_rate > 0, "sample rate must be positive");

    let mono = downmix_to_mono(&recording.samples, recording.channels)?;
    let samples = resample(&mono, recording.sample_rate, STT_SAMPLE_RATE)?;

    Ok(SttAudio { samples })
}

fn downmix_to_mono(interleaved: &[i16], channels: u16) -> Result<Vec<f32>> {
    let channels = channels as usize;
    ensure!(channels > 0, "channel count must be positive");
    ensure!(
        interleaved.len().is_multiple_of(channels),
        "sample buffer contains an incomplete audio frame"
    );

    let scale = 1.0 / (i16::MAX as f32 + 1.0);
    let channel_scale = scale / channels as f32;

    Ok(interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().map(|&sample| sample as f32).sum::<f32>() * channel_scale)
        .collect())
}

fn resample(input: &[f32], input_rate: u32, output_rate: u32) -> Result<Vec<f32>> {
    ensure!(input_rate > 0, "input sample rate must be positive");
    ensure!(output_rate > 0, "output sample rate must be positive");

    if input.is_empty() || input_rate == output_rate {
        return Ok(input.to_vec());
    }

    let input_buffer = InterleavedSlice::new(input, 1, input.len())?;
    let mut resampler = Fft::<f32>::new(
        input_rate as usize,
        output_rate as usize,
        RESAMPLER_CHUNK_SIZE,
        1,
        FixedSync::Both,
    )?;
    let output = resampler.process_all(&input_buffer, input.len(), None)?;

    Ok(output.take_data())
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use super::*;

    #[test]
    fn downmixes_inter_to_mono_and_normalizes() {
        let stereo = [i16::MAX, i16::MIN, 16_384, 16_384];

        let mono = downmix_to_mono(&stereo, 2).unwrap();

        assert_eq!(mono.len(), 2);
        assert!(mono[0].abs() < 0.000_1);
        assert!((mono[1] - 0.5).abs() < 0.000_1);
    }

    #[test]
    fn rejects_incomplete_interleaved_frame() {
        let error = downmix_to_mono(&[1, 2, 3], 2).unwrap_err();

        assert!(error.to_string().contains("incomplete audio frame"));
    }

    #[test]
    fn keeps_samples_when_rate_already_matches_stt_contract() {
        let input = [-1.0, -0.25, 0.0, 0.25, 0.75];

        let output = resample(&input, STT_SAMPLE_RATE, STT_SAMPLE_RATE).unwrap();

        assert_eq!(output, input);
    }

    #[test]
    fn resampling_preserves_one_second_duration() {
        const INPUT_RATE: u32 = 48_000;
        const FREQUENCY: f32 = 1_000.0;
        let input = (0..INPUT_RATE)
            .map(|sample| (TAU * FREQUENCY * sample as f32 / INPUT_RATE as f32).sin())
            .collect::<Vec<_>>();

        let output = resample(&input, INPUT_RATE, STT_SAMPLE_RATE).unwrap();

        assert_eq!(output.len(), STT_SAMPLE_RATE as usize);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }
}
