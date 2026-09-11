use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub type SparseVector = Vec<(usize, f64)>;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct FeatureConfig {
    pub minimum_ngram: usize,
    pub maximum_ngram: usize,
    pub minimum_document_frequency: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            minimum_ngram: 2,
            maximum_ngram: 5,
            minimum_document_frequency: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TfidfVectorizer {
    pub config: FeatureConfig,
    pub vocabulary: BTreeMap<String, usize>,
    pub inverse_document_frequency: Vec<f64>,
}

impl TfidfVectorizer {
    pub fn fit(texts: &[String], config: FeatureConfig) -> Self {
        assert!(config.minimum_ngram > 0);
        assert!(config.minimum_ngram <= config.maximum_ngram);
        assert!(config.minimum_document_frequency > 0);

        let mut document_frequency = BTreeMap::<String, usize>::new();
        for text in texts {
            let unique = character_ngrams(text, config)
                .into_iter()
                .collect::<BTreeSet<_>>();
            for feature in unique {
                *document_frequency.entry(feature).or_default() += 1;
            }
        }

        let vocabulary = document_frequency
            .iter()
            .filter(|(_, frequency)| **frequency >= config.minimum_document_frequency)
            .enumerate()
            .map(|(index, (feature, _))| (feature.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let documents = texts.len() as f64;
        let mut inverse_document_frequency = vec![0.0; vocabulary.len()];
        for (feature, &index) in &vocabulary {
            let frequency = document_frequency[feature] as f64;
            // Smoothed IDF: ln((N + 1) / (DF + 1)) + 1.
            inverse_document_frequency[index] = ((documents + 1.0) / (frequency + 1.0)).ln() + 1.0;
        }

        Self {
            config,
            vocabulary,
            inverse_document_frequency,
        }
    }

    pub fn dimensions(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn transform(&self, text: &str) -> SparseVector {
        let ngrams = character_ngrams(text, self.config);
        if ngrams.is_empty() {
            return Vec::new();
        }

        let total = ngrams.len() as f64;
        let mut counts = BTreeMap::<usize, usize>::new();
        for ngram in ngrams {
            if let Some(&index) = self.vocabulary.get(&ngram) {
                *counts.entry(index).or_default() += 1;
            }
        }

        let mut vector = counts
            .into_iter()
            .map(|(index, count)| {
                // TF is the feature count divided by all extracted n-grams in this document.
                let term_frequency = count as f64 / total;
                (
                    index,
                    term_frequency * self.inverse_document_frequency[index],
                )
            })
            .collect::<SparseVector>();
        let norm = vector
            .iter()
            .map(|(_, value)| value * value)
            .sum::<f64>()
            .sqrt();
        // x = v / ||v||_2 keeps phrase length from dominating the classifier.
        if norm > 0.0 {
            for (_, value) in &mut vector {
                *value /= norm;
            }
        }
        vector
    }
}

pub fn normalize(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn character_ngrams(text: &str, config: FeatureConfig) -> Vec<String> {
    let normalized = normalize(text);
    if normalized.is_empty() {
        return Vec::new();
    }
    let characters = format!("^{normalized}$").chars().collect::<Vec<_>>();
    let mut features = Vec::new();
    for size in config.minimum_ngram..=config.maximum_ngram {
        if size > characters.len() {
            continue;
        }
        for start in 0..=characters.len() - size {
            features.push(characters[start..start + size].iter().collect());
        }
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_unicode_case_punctuation_and_spaces() {
        assert_eq!(normalize("  Открой,  Spotify! "), "открой spotify");
    }

    #[test]
    fn fits_a_stable_train_only_vocabulary_and_l2_normalizes_vectors() {
        let texts = vec!["час".to_owned(), "таймер".to_owned(), "час".to_owned()];
        let config = FeatureConfig {
            minimum_ngram: 2,
            maximum_ngram: 2,
            minimum_document_frequency: 2,
        };
        let first = TfidfVectorizer::fit(&texts, config);
        let second = TfidfVectorizer::fit(&texts, config);

        assert_eq!(first, second);
        assert!(first.vocabulary.contains_key("ча"));
        assert!(!first.vocabulary.contains_key("та"));
        let vector = first.transform("час");
        let norm = vector
            .iter()
            .map(|(_, value)| value * value)
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn idf_downweights_features_seen_in_more_documents() {
        let texts = vec!["час".to_owned(), "часик".to_owned(), "таймер".to_owned()];
        let vectorizer = TfidfVectorizer::fit(
            &texts,
            FeatureConfig {
                minimum_ngram: 2,
                maximum_ngram: 2,
                minimum_document_frequency: 1,
            },
        );

        let common = vectorizer.inverse_document_frequency[vectorizer.vocabulary["ча"]];
        let rare = vectorizer.inverse_document_frequency[vectorizer.vocabulary["та"]];
        assert!(common < rare);
    }
}
