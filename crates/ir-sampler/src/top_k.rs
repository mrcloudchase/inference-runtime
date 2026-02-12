use crate::sampler::{Sampler, TokenLogit};

/// Keeps only the top K tokens by logit value, discarding the rest.
pub struct TopKSampler {
    k: usize,
}

impl TopKSampler {
    /// Create a new top-K sampler that retains the `k` highest-logit tokens.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Sampler for TopKSampler {
    fn name(&self) -> &str {
        "top_k"
    }

    fn apply(&self, logits: &mut Vec<TokenLogit>) {
        if self.k == 0 || self.k >= logits.len() {
            return;
        }

        // Sort descending by logit value.
        logits.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only the top K entries.
        logits.truncate(self.k);
    }
}
