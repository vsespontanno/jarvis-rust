use std::collections::BTreeSet;

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

use super::features::SparseVector;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub l2_regularization: f64,
    pub reject_threshold: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 400,
            learning_rate: 0.5,
            l2_regularization: 0.0001,
            reject_threshold: 0.60,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LogisticRegression {
    pub labels: Vec<String>,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub reject_threshold: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub intent: String,
    pub confidence: f64,
    pub probabilities: Vec<(String, f64)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingStats {
    pub initial_loss: f64,
    pub final_loss: f64,
}

impl LogisticRegression {
    pub fn train(
        features: &[SparseVector],
        targets: &[String],
        dimensions: usize,
        config: TrainingConfig,
    ) -> Result<(Self, TrainingStats)> {
        ensure!(!features.is_empty(), "cannot train on an empty dataset");
        ensure!(
            features.len() == targets.len(),
            "feature/target length mismatch"
        );
        ensure!(dimensions > 0, "cannot train without features");
        ensure!(config.epochs > 0, "epochs must be positive");
        ensure!(config.learning_rate > 0.0, "learning rate must be positive");
        ensure!(config.l2_regularization >= 0.0, "L2 must be non-negative");
        ensure!(
            (0.0..=1.0).contains(&config.reject_threshold),
            "reject threshold must be between zero and one"
        );

        let labels = targets
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        ensure!(labels.len() >= 2, "training requires at least two classes");
        ensure!(
            labels.iter().any(|label| label == "unknown"),
            "training requires the unknown class"
        );
        for vector in features {
            if vector.iter().any(|(index, _)| *index >= dimensions) {
                bail!("feature index exceeds vectorizer dimensions");
            }
        }

        let class_count = labels.len();
        let mut model = Self {
            labels,
            weights: vec![vec![0.0; dimensions]; class_count],
            biases: vec![0.0; class_count],
            reject_threshold: config.reject_threshold,
        };
        let target_indices = targets
            .iter()
            .map(|target| {
                model
                    .labels
                    .binary_search(target)
                    .expect("labels were collected from targets")
            })
            .collect::<Vec<_>>();
        let initial_loss = model.loss(features, &target_indices, config.l2_regularization);

        for _ in 0..config.epochs {
            let mut weight_gradients = vec![vec![0.0; dimensions]; class_count];
            let mut bias_gradients = vec![0.0; class_count];
            for (vector, &target) in features.iter().zip(&target_indices) {
                let probabilities = model.probabilities(vector);
                for class in 0..class_count {
                    // dJ/dz_c = p_c - 1[y = c] for softmax cross-entropy.
                    let error = probabilities[class] - f64::from(class == target);
                    bias_gradients[class] += error;
                    for &(feature, value) in vector {
                        weight_gradients[class][feature] += error * value;
                    }
                }
            }

            let samples = features.len() as f64;
            for class in 0..class_count {
                model.biases[class] -= config.learning_rate * bias_gradients[class] / samples;
                for (feature, (weight, gradient)) in model.weights[class]
                    .iter_mut()
                    .zip(&weight_gradients[class])
                    .enumerate()
                {
                    debug_assert!(feature < dimensions);
                    let gradient = gradient / samples + config.l2_regularization * *weight;
                    // Full-batch gradient descent: w <- w - learning_rate * dJ/dw.
                    *weight -= config.learning_rate * gradient;
                }
            }
        }

        let final_loss = model.loss(features, &target_indices, config.l2_regularization);
        Ok((
            model,
            TrainingStats {
                initial_loss,
                final_loss,
            },
        ))
    }

    pub fn predict(&self, features: &SparseVector) -> Prediction {
        let probabilities = self.probabilities(features);
        let (best_class, &confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .expect("a trained model has classes");
        let raw_intent = &self.labels[best_class];
        let intent = if confidence < self.reject_threshold {
            "unknown"
        } else {
            raw_intent
        };

        Prediction {
            intent: intent.to_owned(),
            confidence,
            probabilities: self.labels.iter().cloned().zip(probabilities).collect(),
        }
    }

    fn probabilities(&self, features: &SparseVector) -> Vec<f64> {
        let mut scores = self
            .weights
            .iter()
            .zip(&self.biases)
            .map(|(weights, bias)| {
                bias + features
                    .iter()
                    .map(|(feature, value)| weights[*feature] * value)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Stable softmax subtracts max(z) before exponentiation.
        for score in &mut scores {
            *score = (*score - maximum).exp();
        }
        let total = scores.iter().sum::<f64>();
        for score in &mut scores {
            *score /= total;
        }
        scores
    }

    fn loss(&self, features: &[SparseVector], targets: &[usize], l2: f64) -> f64 {
        let cross_entropy = features
            .iter()
            .zip(targets)
            .map(|(vector, &target)| -self.probabilities(vector)[target].max(1e-15).ln())
            .sum::<f64>()
            / features.len() as f64;
        let penalty = 0.5
            * l2
            * self
                .weights
                .iter()
                .flat_map(|weights| weights.iter())
                .map(|weight| weight * weight)
                .sum::<f64>();
        cross_entropy + penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_stable_and_probabilities_sum_to_one() {
        let model = LogisticRegression {
            labels: vec!["tell_time".to_owned(), "unknown".to_owned()],
            weights: vec![vec![1_000.0], vec![-1_000.0]],
            biases: vec![1_000.0, -1_000.0],
            reject_threshold: 0.0,
        };

        let probabilities = model.probabilities(&vec![(0, 1.0)]);
        assert!(probabilities.iter().all(|value| value.is_finite()));
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gradient_descent_reduces_loss_and_learns_a_toy_problem() {
        let features = vec![
            vec![(0, 1.0)],
            vec![(0, 0.9)],
            vec![(1, 1.0)],
            vec![(1, 0.9)],
        ];
        let targets = vec![
            "tell_time".to_owned(),
            "tell_time".to_owned(),
            "unknown".to_owned(),
            "unknown".to_owned(),
        ];
        let (model, stats) = LogisticRegression::train(
            &features,
            &targets,
            2,
            TrainingConfig {
                epochs: 200,
                reject_threshold: 0.5,
                ..TrainingConfig::default()
            },
        )
        .unwrap();

        assert!(stats.final_loss < stats.initial_loss);
        assert_eq!(model.predict(&vec![(0, 1.0)]).intent, "tell_time");
        assert_eq!(model.predict(&vec![(1, 1.0)]).intent, "unknown");
    }

    #[test]
    fn low_confidence_prediction_is_safely_rejected() {
        let model = LogisticRegression {
            labels: vec!["tell_time".to_owned(), "unknown".to_owned()],
            weights: vec![vec![0.0], vec![0.0]],
            biases: vec![0.0, 0.0],
            reject_threshold: 0.75,
        };

        let prediction = model.predict(&Vec::new());
        assert_eq!(prediction.intent, "unknown");
        assert_eq!(prediction.confidence, 0.5);
    }
}
