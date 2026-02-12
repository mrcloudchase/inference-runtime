use std::collections::HashMap;

use crate::error::{ModelError, Result};
use crate::gguf::metadata::GgufMetadata;
use super::vocab::Vocab;

/// Byte-Pair Encoding tokenizer loaded from GGUF metadata.
pub struct BpeTokenizer {
    /// The token vocabulary (strings, scores, special token IDs).
    pub vocab: Vocab,
    /// Ordered merge rules. Each entry is a pair of token strings that can be
    /// merged. Earlier entries have higher priority. Retained for inspection
    /// and potential serialization.
    #[allow(dead_code)]
    merges: Vec<(String, String)>,
    /// Map from merge pair to priority rank (lower rank = higher priority).
    merge_ranks: HashMap<(String, String), usize>,
}

impl BpeTokenizer {
    /// Load a BPE tokenizer from GGUF metadata.
    ///
    /// Reads the vocabulary via `Vocab::from_gguf`, then loads merge rules
    /// from the `tokenizer.ggml.merges` metadata key (a string array where
    /// each entry is "token1 token2").
    pub fn from_gguf(metadata: &GgufMetadata) -> Result<BpeTokenizer> {
        let vocab = Vocab::from_gguf(metadata)?;

        let merge_strings = metadata
            .get_string_array("tokenizer.ggml.merges")
            .unwrap_or_default();

        let mut merges = Vec::with_capacity(merge_strings.len());
        let mut merge_ranks = HashMap::with_capacity(merge_strings.len());

        for (rank, entry) in merge_strings.iter().enumerate() {
            // Each merge entry is "token1 token2" separated by a single space.
            // Split on the first space only.
            let parts: Vec<&str> = entry.splitn(2, ' ').collect();
            if parts.len() != 2 {
                return Err(ModelError::TokenizerError(format!(
                    "invalid merge entry: {:?}",
                    entry
                )));
            }
            let pair = (parts[0].to_string(), parts[1].to_string());
            merge_ranks.insert(pair.clone(), rank);
            merges.push(pair);
        }

        Ok(BpeTokenizer {
            vocab,
            merges,
            merge_ranks,
        })
    }

    /// Encode a text string into a sequence of token IDs using BPE.
    ///
    /// Algorithm:
    /// 1. Convert the input text to individual UTF-8 bytes.
    /// 2. Map each byte to the corresponding byte-level token in the vocabulary.
    ///    Byte tokens are stored as `<0xHH>` where HH is the hex value, or as
    ///    the literal character if it appears that way in the vocab.
    /// 3. Iteratively find and apply the highest-priority merge pair until no
    ///    more merges can be applied.
    /// 4. Convert the resulting token strings to IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with individual characters as tokens. For each byte, try to
        // find it in the vocabulary (either as a single character or as a
        // byte-level token like <0x41>).
        let mut tokens: Vec<String> = Vec::new();

        for byte in text.bytes() {
            // Try the byte as a single character first.
            let ch = byte as char;
            let ch_str = ch.to_string();
            if self.vocab.token_to_id.contains_key(&ch_str) {
                tokens.push(ch_str);
            } else {
                // Try byte-level token format: <0xHH>
                let byte_token = format!("<0x{:02X}>", byte);
                if self.vocab.token_to_id.contains_key(&byte_token) {
                    tokens.push(byte_token);
                } else {
                    // Fallback: use the character string anyway; it will map
                    // to an unknown token at the end.
                    tokens.push(ch_str);
                }
            }
        }

        // Iteratively apply BPE merges.
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the best (lowest rank) merge pair among all adjacent pairs.
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                // No more merges possible.
                break;
            }

            // Merge the pair at best_idx.
            let merged = format!("{}{}", tokens[best_idx], tokens[best_idx + 1]);
            tokens[best_idx] = merged;
            tokens.remove(best_idx + 1);
        }

        // Convert token strings to IDs.
        tokens
            .iter()
            .map(|tok| {
                self.vocab
                    .token_to_id
                    .get(tok)
                    .copied()
                    .unwrap_or(0) // fallback to token 0 for unknown tokens
            })
            .collect()
    }

    /// Decode a sequence of token IDs back into a string.
    ///
    /// Maps each ID to its token string and concatenates. Byte-level tokens
    /// of the form `<0xHH>` are converted back to the corresponding byte.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();

        for &id in tokens {
            let id = id as usize;
            if id >= self.vocab.tokens.len() {
                continue;
            }
            let tok = &self.vocab.tokens[id];

            // Check if this is a byte-level token like <0xHH>.
            if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(&tok[3..5], 16) {
                    bytes.push(byte_val);
                    continue;
                }
            }

            // Otherwise, append the token's UTF-8 bytes directly.
            bytes.extend_from_slice(tok.as_bytes());
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Returns the beginning-of-sequence token ID.
    pub fn bos_id(&self) -> u32 {
        self.vocab.bos_id
    }

    /// Returns the end-of-sequence token ID.
    pub fn eos_id(&self) -> u32 {
        self.vocab.eos_id
    }
}
