use std::collections::HashMap;

use crate::error::{ModelError, Result};
use crate::gguf::metadata::GgufMetadata;

/// Token vocabulary loaded from GGUF metadata.
pub struct Vocab {
    /// Token strings, indexed by token ID.
    pub tokens: Vec<String>,
    /// Merge priority scores, indexed by token ID.
    pub scores: Vec<f32>,
    /// Reverse mapping from token string to token ID.
    pub token_to_id: HashMap<String, u32>,
    /// Beginning-of-sequence token ID.
    pub bos_id: u32,
    /// End-of-sequence token ID.
    pub eos_id: u32,
}

impl Vocab {
    /// Build a vocabulary from GGUF metadata.
    ///
    /// Reads the following metadata keys:
    /// - `tokenizer.ggml.tokens` (string array of token strings)
    /// - `tokenizer.ggml.scores` (f32 array of merge scores)
    /// - `tokenizer.ggml.bos_token_id` (u32)
    /// - `tokenizer.ggml.eos_token_id` (u32)
    pub fn from_gguf(metadata: &GgufMetadata) -> Result<Vocab> {
        let tokens = metadata.get_string_array("tokenizer.ggml.tokens")?;
        let scores = metadata.get_f32_array("tokenizer.ggml.scores")?;

        if tokens.len() != scores.len() {
            return Err(ModelError::TokenizerError(format!(
                "tokens length ({}) does not match scores length ({})",
                tokens.len(),
                scores.len()
            )));
        }

        let bos_id = metadata.get_u32("tokenizer.ggml.bos_token_id")?;
        let eos_id = metadata.get_u32("tokenizer.ggml.eos_token_id")?;

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
        }

        Ok(Vocab {
            tokens,
            scores,
            token_to_id,
            bos_id,
            eos_id,
        })
    }

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
