use crate::audio::AudioLevel;

#[derive(Clone, Copy, Debug)]
pub struct VadConfig {
    pub calibration_windows: usize,
    pub speech_start_windows: usize,
    pub speech_end_windows: usize,
    pub max_wait_windows: usize,
    pub max_speech_windows: usize,
    pub start_margin_db: f32,
    pub end_margin_db: f32,
    pub noise_ema_alpha: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechEndReason {
    Silence,
    MaximumDuration,
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
        loud_windows: usize,
        waited: usize,
    },
    Speaking {
        quiet_windows: usize,
        elapsed: usize,
    },
    Finished,
}

#[derive(Debug)]
pub struct VadDetector {
    config: VadConfig,
    state: State,
    noise_floor_dbfs: Option<f32>,
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

        Self {
            config,
            state: State::Calibrating {
                levels: Vec::with_capacity(config.calibration_windows),
            },
            noise_floor_dbfs: None,
        }
    }

    pub fn observe(&mut self, level: AudioLevel) -> Option<VadEvent> {
        match &mut self.state {
            State::Calibrating { levels } => {
                levels.push(level.dbfs);
                if levels.len() < self.config.calibration_windows {
                    return None;
                }

                let noise_floor_dbfs = median(levels);
                self.noise_floor_dbfs = Some(noise_floor_dbfs);
                self.state = State::Waiting {
                    loud_windows: 0,
                    waited: 0,
                };
                Some(VadEvent::Calibrated { noise_floor_dbfs })
            }
            State::Waiting {
                loud_windows,
                waited,
            } => {
                *waited += 1;
                let noise_floor = self
                    .noise_floor_dbfs
                    .expect("noise floor is set after calibration");
                let start_threshold = noise_floor + self.config.start_margin_db;

                if level.dbfs >= start_threshold {
                    *loud_windows += 1;
                } else {
                    *loud_windows = 0;
                    self.noise_floor_dbfs = Some(
                        (1.0 - self.config.noise_ema_alpha) * noise_floor
                            + self.config.noise_ema_alpha * level.dbfs,
                    );
                }

                if *loud_windows >= self.config.speech_start_windows {
                    self.state = State::Speaking {
                        quiet_windows: 0,
                        elapsed: 0,
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
            } => {
                *elapsed += 1;
                let end_threshold = self
                    .noise_floor_dbfs
                    .expect("noise floor is set after calibration")
                    + self.config.end_margin_db;

                if level.dbfs < end_threshold {
                    *quiet_windows += 1;
                } else {
                    *quiet_windows = 0;
                }

                let reason = if *quiet_windows >= self.config.speech_end_windows {
                    Some(SpeechEndReason::Silence)
                } else if *elapsed >= self.config.max_speech_windows {
                    Some(SpeechEndReason::MaximumDuration)
                } else {
                    None
                };

                reason.map(|reason| {
                    self.state = State::Finished;
                    VadEvent::SpeechEnded {
                        at_frame: level.end_frame,
                        reason,
                    }
                })
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
}
