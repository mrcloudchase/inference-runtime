use crate::sampler::{Sampler, TokenLogit};

/// Applies a repetition penalty to tokens that have appeared recently.
///
/// For tokens found in `recent_tokens`:
/// - Positive logits are divided by `penalty`.
/// - Negative logits are multiplied by `penalty`.
///
/// This discourages the model from repeating the same tokens.
pub struct RepetitionPenaltySampler {
    penalty: f32,
    recent_tokens: Vec<u32>,
    max_history: usize,
}

impl RepetitionPenaltySampler {
    /// Create a new repetition penalty sampler.
    ///
    /// - `penalty`: the penalty factor (1.0 = no penalty).
    /// - `max_history`: maximum number of recent tokens to track.
    pub fn new(penalty: f32, max_history: usize) -> Self {
        Self {
            penalty,
            recent_tokens: Vec::new(),
            max_history,
        }
    }

    /// Record a generated token so it will be penalized in future sampling steps.
    pub fn add_token(&mut self, token: u32) {
        self.recent_tokens.push(token);
        if self.recent_tokens.len() > self.max_history {
            self.recent_tokens.remove(0);
        }
    }
}

impl Sampler for RepetitionPenaltySampler {
    fn name(&self) -> &str {
        "repetition_penalty"
    }

    fn apply(&self, logits: &mut Vec<TokenLogit>) {
        for token in logits.iter_mut() {
            if self.recent_tokens.contains(&token.token_id) {
                if token.logit > 0.0 {
                    token.logit /= self.penalty;
                } else {
                    token.logit *= self.penalty;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.recent_tokens.clear();
    }
}
