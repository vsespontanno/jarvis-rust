use crate::audio::AudioLevel;

const DIGITAL_SILENCE_CUTOFF_DBFS: f32 = -120.0;
const FALLBACK_NOISE_FLOOR_DBFS: f32 = -90.0;

#[derive(Clone, Copy, Debug)]
pub struct VadConfig {
    pub calibration_windows: usize,
    pub speech_start_windows: usize,
    pub speech_end_windows: usize,
    pub max_wait_windows: usize,
    pub max_speech_windows: usize,
    pub start_margin_db: f32,
    pub end_margin_db: f32,
    pub minimum_start_level_dbfs: f32,
    pub minimum_end_level_dbfs: f32,
    pub noise_ema_alpha: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechEndReason {
    Silence,
    MaximumDuration,
}

impl SpeechEndReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Silence => "silence",
            Self::MaximumDuration => "maximum_duration",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadMetrics {
    pub noise_floor_dbfs: f32,
    pub start_threshold_dbfs: f32,
    pub end_threshold_dbfs: f32,
    pub peak_dbfs: f32,
    pub mean_speech_dbfs: f32,
    pub median_speech_dbfs: f32,
    pub detected_windows: usize,
    pub trailing_silence_windows: usize,
    pub speech_windows: usize,
    pub silence_windows: usize,
    pub end_reason: SpeechEndReason,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VadEvent {
    Calibrated {
        noise_floor_dbfs: f32,
    },
    SpeechStarted {
        at_frame: usize,
    },
    SpeechEnded {
        at_frame: usize,
        reason: SpeechEndReason,
    },
    TimedOut,
}

#[derive(Debug)]
enum State {
    Calibrating {
        levels: Vec<f32>,
    },
    Waiting {
        loud_levels: Vec<f32>,
        waited: usize,
    },
    Speaking {
        quiet_windows: usize,
        elapsed: usize,
        speech_levels: Vec<f32>,
        silence_windows: usize,
        peak_dbfs: f32,
    },
    Finished,
}

#[derive(Debug)]
pub struct VadDetector {
    config: VadConfig,
    state: State,
    noise_floor_dbfs: Option<f32>,
    metrics: Option<VadMetrics>,
}

impl VadDetector {
    pub fn new(config: VadConfig) -> Self {
        assert!(config.calibration_windows > 0);
        assert!(config.speech_start_windows > 0);
        assert!(config.speech_end_windows > 0);
        assert!(config.max_wait_windows > 0);
        assert!(config.max_speech_windows > 0);
        assert!((0.0..=1.0).contains(&config.noise_ema_alpha));
        assert!(config.start_margin_db > config.end_margin_db);
        assert!(config.minimum_start_level_dbfs > config.minimum_end_level_dbfs);

        Self {
            config,
            state: State::Calibrating {
                levels: Vec::with_capacity(config.calibration_windows),
            },
            noise_floor_dbfs: None,
            metrics: None,
        }
    }

    pub fn metrics(&self) -> Option<VadMetrics> {
        self.metrics
    }

    pub fn is_calibrated(&self) -> bool {
        self.noise_floor_dbfs.is_some()
    }

    pub fn noise_floor_dbfs(&self) -> Option<f32> {
        self.noise_floor_dbfs
    }

    pub fn begin_utterance(&mut self) {
        self.metrics = None;
        self.state = if self.noise_floor_dbfs.is_some() {
            State::Waiting {
                loud_levels: Vec::with_capacity(self.config.speech_start_windows),
                waited: 0,
            }
        } else {
            State::Calibrating {
                levels: Vec::with_capacity(self.config.calibration_windows),
            }
        };
    }

    pub fn observe(&mut self, level: AudioLevel) -> Option<VadEvent> {
        match &mut self.state {
            State::Calibrating { levels } => {
                levels.push(level.dbfs);
                if levels.len() < self.config.calibration_windows {
                    return None;
                }

                let noise_floor_dbfs = estimate_noise_floor(levels);
                self.noise_floor_dbfs = Some(noise_floor_dbfs);
                self.state = State::Waiting {
                    loud_levels: Vec::with_capacity(self.config.speech_start_windows),
                    waited: 0,
                };
                Some(VadEvent::Calibrated { noise_floor_dbfs })
            }
            State::Waiting {
                loud_levels,
                waited,
            } => {
                *waited += 1;
                let noise_floor = self
                    .noise_floor_dbfs
                    .expect("noise floor is set after calibration");
                let start_threshold = (noise_floor + self.config.start_margin_db)
                    .max(self.config.minimum_start_level_dbfs);

                if level.dbfs >= start_threshold {
                    loud_levels.push(level.dbfs);
                } else {
                    loud_levels.clear();
                    if level.dbfs > DIGITAL_SILENCE_CUTOFF_DBFS {
                        self.noise_floor_dbfs = Some(
                            (1.0 - self.config.noise_ema_alpha) * noise_floor
                                + self.config.noise_ema_alpha * level.dbfs,
                        );
                    }
                }

                if loud_levels.len() >= self.config.speech_start_windows {
                    let speech_levels = std::mem::take(loud_levels);
                    let peak_dbfs = speech_levels
                        .iter()
                        .copied()
                        .max_by(f32::total_cmp)
                        .expect("speech start contains at least one level");
                    self.state = State::Speaking {
                        quiet_windows: 0,
                        elapsed: 0,
                        speech_levels,
                        silence_windows: 0,
                        peak_dbfs,
                    };
                    return Some(VadEvent::SpeechStarted {
                        at_frame: level.end_frame,
                    });
                }

                if *waited >= self.config.max_wait_windows {
                    self.state = State::Finished;
                    return Some(VadEvent::TimedOut);
                }

                None
            }
            State::Speaking {
                quiet_windows,
                elapsed,
                speech_levels,
                silence_windows,
                peak_dbfs,
            } => {
                *elapsed += 1;
                let end_threshold = (self
                    .noise_floor_dbfs
                    .expect("noise floor is set after calibration")
                    + self.config.end_margin_db)
                    .max(self.config.minimum_end_level_dbfs);

                if level.dbfs < end_threshold {
                    *quiet_windows += 1;
                    *silence_windows += 1;
                } else {
                    *quiet_windows = 0;
                    speech_levels.push(level.dbfs);
                }
                *peak_dbfs = peak_dbfs.max(level.dbfs);

                let reason = if *quiet_windows >= self.config.speech_end_windows {
                    Some(SpeechEndReason::Silence)
                } else if *elapsed >= self.config.max_speech_windows {
                    Some(SpeechEndReason::MaximumDuration)
                } else {
                    None
                };

                if let Some(reason) = reason {
                    let mean_speech_dbfs =
                        speech_levels.iter().sum::<f32>() / speech_levels.len() as f32;
                    let median_speech_dbfs = median(speech_levels);
                    let noise_floor_dbfs = self
                        .noise_floor_dbfs
                        .expect("noise floor is set after calibration");
                    let metrics = VadMetrics {
                        noise_floor_dbfs,
                        start_threshold_dbfs: (noise_floor_dbfs + self.config.start_margin_db)
                            .max(self.config.minimum_start_level_dbfs),
                        end_threshold_dbfs: end_threshold,
                        peak_dbfs: *peak_dbfs,
                        mean_speech_dbfs,
                        median_speech_dbfs,
                        detected_windows: speech_levels.len() + *silence_windows,
                        trailing_silence_windows: *quiet_windows,
                        speech_windows: speech_levels.len(),
                        silence_windows: *silence_windows,
                        end_reason: reason,
                    };
                    self.metrics = Some(metrics);
                    self.state = State::Finished;
                    Some(VadEvent::SpeechEnded {
                        at_frame: level.end_frame,
                        reason,
                    })
                } else {
                    None
                }
            }
            State::Finished => None,
        }
    }
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;

    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn estimate_noise_floor(levels: &mut Vec<f32>) -> f32 {
    levels.retain(|level| *level > DIGITAL_SILENCE_CUTOFF_DBFS);

    if levels.is_empty() {
        FALLBACK_NOISE_FLOOR_DBFS
    } else {
        median(levels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VadConfig {
        VadConfig {
            calibration_windows: 3,
            speech_start_windows: 2,
            speech_end_windows: 2,
            max_wait_windows: 4,
            max_speech_windows: 5,
            start_margin_db: 12.0,
            end_margin_db: 6.0,
            minimum_start_level_dbfs: -100.0,
            minimum_end_level_dbfs: -110.0,
            noise_ema_alpha: 0.0,
        }
    }

    fn level(dbfs: f32, end_frame: usize) -> AudioLevel {
        AudioLevel {
            rms: 10.0f32.powf(dbfs / 20.0),
            dbfs,
            end_frame,
        }
    }

    fn calibrated_detector() -> VadDetector {
        let mut detector = VadDetector::new(config());
        assert!(detector.observe(level(-51.0, 1)).is_none());
        assert!(detector.observe(level(-50.0, 2)).is_none());
        assert_eq!(
            detector.observe(level(-49.0, 3)),
            Some(VadEvent::Calibrated {
                noise_floor_dbfs: -50.0
            })
        );
        detector
    }

    #[test]
    fn requires_consecutive_loud_windows_to_start_speech() {
        let mut detector = calibrated_detector();

        assert!(detector.observe(level(-30.0, 4)).is_none());
        assert_eq!(
            detector.observe(level(-29.0, 5)),
            Some(VadEvent::SpeechStarted { at_frame: 5 })
        );
    }

    #[test]
    fn hysteresis_ignores_short_quiet_gap() {
        let mut detector = calibrated_detector();
        detector.observe(level(-30.0, 4));
        detector.observe(level(-30.0, 5));

        assert!(detector.observe(level(-48.0, 6)).is_none());
        assert!(detector.observe(level(-30.0, 7)).is_none());
        assert!(detector.observe(level(-48.0, 8)).is_none());
        assert_eq!(
            detector.observe(level(-48.0, 9)),
            Some(VadEvent::SpeechEnded {
                at_frame: 9,
                reason: SpeechEndReason::Silence,
            })
        );
    }

    #[test]
    fn times_out_when_no_speech_arrives() {
        let mut detector = calibrated_detector();

        for frame in 4..7 {
            assert!(detector.observe(level(-50.0, frame)).is_none());
        }
        assert_eq!(detector.observe(level(-50.0, 7)), Some(VadEvent::TimedOut));
    }

    #[test]
    fn stops_an_overlong_utterance() {
        let mut detector = calibrated_detector();
        detector.observe(level(-30.0, 4));
        detector.observe(level(-30.0, 5));

        for frame in 6..10 {
            assert!(detector.observe(level(-30.0, frame)).is_none());
        }
        assert_eq!(
            detector.observe(level(-30.0, 10)),
            Some(VadEvent::SpeechEnded {
                at_frame: 10,
                reason: SpeechEndReason::MaximumDuration,
            })
        );
    }

    #[test]
    fn ignores_digitally_gated_silence_during_calibration() {
        let mut detector = VadDetector::new(config());

        detector.observe(level(-240.0, 1));
        detector.observe(level(-240.0, 2));

        assert_eq!(
            detector.observe(level(-50.0, 3)),
            Some(VadEvent::Calibrated {
                noise_floor_dbfs: -50.0,
            })
        );
    }

    #[test]
    fn absolute_threshold_rejects_low_level_background_speech() {
        let mut config = config();
        config.minimum_start_level_dbfs = -35.0;
        config.minimum_end_level_dbfs = -40.0;
        let mut detector = VadDetector::new(config);
        detector.observe(level(-51.0, 1));
        detector.observe(level(-50.0, 2));
        detector.observe(level(-49.0, 3));

        assert!(detector.observe(level(-40.0, 4)).is_none());
        assert!(detector.observe(level(-40.0, 5)).is_none());
        assert!(detector.observe(level(-30.0, 6)).is_none());
        assert_eq!(
            detector.observe(level(-30.0, 7)),
            Some(VadEvent::SpeechStarted { at_frame: 7 })
        );
        assert!(detector.observe(level(-42.0, 8)).is_none());
        assert_eq!(
            detector.observe(level(-42.0, 9)),
            Some(VadEvent::SpeechEnded {
                at_frame: 9,
                reason: SpeechEndReason::Silence,
            })
        );
    }

    #[test]
    fn aggregates_completed_utterance_metrics() {
        let mut detector = calibrated_detector();
        detector.observe(level(-30.0, 4));
        detector.observe(level(-30.0, 5));
        detector.observe(level(-28.0, 6));
        detector.observe(level(-48.0, 7));
        detector.observe(level(-48.0, 8));

        let metrics = detector.metrics().unwrap();
        assert_eq!(metrics.noise_floor_dbfs, -50.0);
        assert_eq!(metrics.start_threshold_dbfs, -38.0);
        assert_eq!(metrics.end_threshold_dbfs, -44.0);
        assert_eq!(metrics.peak_dbfs, -28.0);
        assert!((metrics.mean_speech_dbfs - -29.333_334).abs() < 0.000_1);
        assert_eq!(metrics.median_speech_dbfs, -30.0);
        assert_eq!(metrics.detected_windows, 5);
        assert_eq!(metrics.trailing_silence_windows, 2);
        assert_eq!(metrics.speech_windows, 3);
        assert_eq!(metrics.silence_windows, 2);
        assert_eq!(metrics.end_reason, SpeechEndReason::Silence);
    }

    #[test]
    fn reuses_calibration_for_the_next_utterance() {
        let mut detector = calibrated_detector();
        detector.observe(level(-30.0, 4));
        detector.observe(level(-30.0, 5));
        detector.observe(level(-48.0, 6));
        detector.observe(level(-48.0, 7));
        assert!(detector.metrics().is_some());

        detector.begin_utterance();

        assert!(detector.is_calibrated());
        assert_eq!(detector.noise_floor_dbfs(), Some(-50.0));
        assert!(detector.metrics().is_none());
        assert!(detector.observe(level(-30.0, 1)).is_none());
        assert_eq!(
            detector.observe(level(-30.0, 2)),
            Some(VadEvent::SpeechStarted { at_frame: 2 })
        );
    }
}
