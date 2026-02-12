use crate::sampler::{Sampler, TokenLogit};
use rand::rngs::StdRng;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;

/// Greedy sampler: selects the single token with the highest logit.
pub struct GreedySampler;

impl GreedySampler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GreedySampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for GreedySampler {
    fn name(&self) -> &str {
        "greedy"
    }

    fn apply(&self, logits: &mut Vec<TokenLogit>) {
        if logits.is_empty() {
            return;
        }

        // Sort descending by logit value.
        logits.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only the top 1.
        logits.truncate(1);
    }
}

/// Distribution-based sampler: converts logits to probabilities via softmax,
/// then samples from the resulting distribution using a seeded RNG.
pub struct DistSampler {
    seed: u64,
}

impl DistSampler {
    /// Create a new distribution sampler with the given seed for reproducibility.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl Sampler for DistSampler {
    fn name(&self) -> &str {
        "dist"
    }

    fn apply(&self, logits: &mut Vec<TokenLogit>) {
        if logits.is_empty() {
            return;
        }

        // Compute softmax probabilities.
        let max_logit = logits
            .iter()
            .map(|t| t.logit)
            .fold(f32::NEG_INFINITY, f32::max);

        let exps: Vec<f32> = logits.iter().map(|t| (t.logit - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Sample from the weighted distribution.
        let mut rng = StdRng::seed_from_u64(self.seed);
        let dist = match WeightedIndex::new(&probs) {
            Ok(d) => d,
            Err(_) => {
                // Fallback: keep only the first token if weights are invalid.
                logits.truncate(1);
                return;
            }
        };

        let selected_index = dist.sample(&mut rng);
        let selected = logits[selected_index].clone();

        logits.clear();
        logits.push(selected);
    }
}
