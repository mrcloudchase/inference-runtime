/// A token ID paired with its logit value.
#[derive(Debug, Clone)]
pub struct TokenLogit {
    pub token_id: u32,
    pub logit: f32,
}

/// Trait for samplers that modify or select from a set of token logits.
pub trait Sampler: Send + Sync {
    /// Returns the name of this sampler.
    fn name(&self) -> &str;

    /// Modify logits in-place (filtering, scaling, etc.)
    fn apply(&self, logits: &mut Vec<TokenLogit>);

    /// Reset any internal state. Default implementation does nothing.
    fn reset(&mut self) {}
}

/// Composes multiple samplers into a pipeline.
/// The last sampler in the chain should be a selector (greedy or random).
pub struct SamplerChain {
    samplers: Vec<Box<dyn Sampler>>,
}

impl SamplerChain {
    /// Create a new empty sampler chain.
    pub fn new() -> Self {
        Self {
            samplers: Vec::new(),
        }
    }

    /// Add a sampler to the end of the chain. Returns self for builder-style usage.
    pub fn with(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.samplers.push(sampler);
        self
    }

    /// Run all samplers in order on raw logits, return the selected token ID.
    ///
    /// 1. Converts the `&[f32]` logits into `Vec<TokenLogit>` (token_id = index).
    /// 2. Applies each sampler in sequence.
    /// 3. Returns the first token's id (the selected one).
    pub fn sample(&self, logits: &[f32]) -> u32 {
        let mut token_logits: Vec<TokenLogit> = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| TokenLogit {
                token_id: i as u32,
                logit,
            })
            .collect();

        for sampler in &self.samplers {
            sampler.apply(&mut token_logits);
        }

        token_logits
            .first()
            .map(|t| t.token_id)
            .unwrap_or(0)
    }
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self::new()
    }
}
