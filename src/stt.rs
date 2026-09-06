use std::path::Path;

use anyhow::{Result, ensure};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Debug)]
pub struct WhisperTranscriber {
    context: WhisperContext,
}

impl WhisperTranscriber {
    pub fn load(model_path: &Path) -> Result<Self> {
        ensure!(
            model_path.is_file(),
            "model file does not exist: {}",
            model_path.display()
        );
        let context =
            WhisperContext::new_with_params(model_path, WhisperContextParameters::default())?;
        ensure!(
            context.is_multilingual(),
            "the selected model is English-only; use a multilingual model without the .en suffix"
        );

        Ok(Self { context })
    }

    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        ensure!(!samples.is_empty(), "cannot transcribe empty audio");

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("ru"));
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_no_timestamps(true);

        let mut state = self.context.create_state()?;
        state.full(params, samples)?;

        let transcript = state
            .as_iter()
            .map(|segment| segment.to_string())
            .collect::<String>()
            .trim()
            .to_owned();

        Ok(transcript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_model_before_entering_ffi() {
        let error = WhisperTranscriber::load(Path::new("models/does-not-exist.bin")).unwrap_err();

        assert!(error.to_string().contains("model file does not exist"));
    }
}
