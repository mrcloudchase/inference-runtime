use crate::sampler::{Sampler, TokenLogit};

/// Scales all logits by dividing by a temperature value.
///
/// Higher temperatures produce more uniform distributions (more random),
/// while lower temperatures sharpen the distribution (more deterministic).
pub struct TemperatureSampler {
    temperature: f32,
}

impl TemperatureSampler {
    /// Create a new temperature sampler with the given temperature.
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl Sampler for TemperatureSampler {
    fn name(&self) -> &str {
        "temperature"
    }

    fn apply(&self, logits: &mut Vec<TokenLogit>) {
        // Clamp temperature to a very small positive value if it is <= 0.
        let temp = if self.temperature <= 0.0 {
            1e-7
        } else {
            self.temperature
        };

        for token in logits.iter_mut() {
            token.logit /= temp;
        }
    }
}
