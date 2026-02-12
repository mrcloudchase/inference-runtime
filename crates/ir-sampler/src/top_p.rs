use crate::sampler::{Sampler, TokenLogit};

/// Nucleus sampling: keeps the smallest set of tokens whose cumulative
/// probability exceeds the threshold `p`.
pub struct TopPSampler {
    p: f32,
}

impl TopPSampler {
    /// Create a new top-p (nucleus) sampler with the given probability threshold.
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl Sampler for TopPSampler {
    fn name(&self) -> &str {
        "top_p"
    }

    fn apply(&self, logits: &mut Vec<TokenLogit>) {
        if logits.is_empty() {
            return;
        }

        // Sort descending by logit value.
        logits.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(std::cmp::Ordering::Equal));

        // Compute softmax probabilities.
        let max_logit = logits[0].logit;
        let exps: Vec<f32> = logits.iter().map(|t| (t.logit - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Find the cutoff index: keep tokens until cumulative probability exceeds p.
        let mut cumulative = 0.0f32;
        let mut cutoff = logits.len();
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative > self.p {
                cutoff = i + 1;
                break;
            }
        }

        // Always keep at least one token.
        if cutoff == 0 {
            cutoff = 1;
        }

        logits.truncate(cutoff);
    }
}
