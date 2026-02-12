/// Key-Value cache for transformer attention layers.
///
/// Stores previously computed key and value projections so they do not need
/// to be recomputed for each new token during autoregressive generation.
///
/// Layout for each layer:
///   k[layer]: flat array of shape [max_seq_len, n_kv_heads * head_dim]
///   v[layer]: flat array of shape [max_seq_len, n_kv_heads * head_dim]
pub struct KvCache {
    /// Key cache for each layer.
    /// k[layer] has size n_kv_heads * max_seq_len * head_dim.
    pub k: Vec<Vec<f32>>,
    /// Value cache for each layer.
    /// v[layer] has size n_kv_heads * max_seq_len * head_dim.
    pub v: Vec<Vec<f32>>,
    /// Number of key/value attention heads.
    pub n_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// Current number of tokens stored in the cache.
    pub len: usize,
}

impl KvCache {
    /// Create a new KV cache with all values initialized to zero.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let cache_size = n_kv_heads * max_seq_len * head_dim;
        let k = (0..n_layers).map(|_| vec![0.0f32; cache_size]).collect();
        let v = (0..n_layers).map(|_| vec![0.0f32; cache_size]).collect();

        KvCache {
            k,
            v,
            n_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Write key and value vectors for one token at a given position in the cache.
    ///
    /// - `layer`: the transformer layer index
    /// - `k_data`: key vector of length n_kv_heads * head_dim
    /// - `v_data`: value vector of length n_kv_heads * head_dim
    /// - `pos`: the sequence position to write at
    pub fn update(&mut self, layer: usize, k_data: &[f32], v_data: &[f32], pos: usize) {
        let kv_dim = self.n_kv_heads * self.head_dim;
        let offset = pos * kv_dim;

        self.k[layer][offset..offset + kv_dim].copy_from_slice(k_data);
        self.v[layer][offset..offset + kv_dim].copy_from_slice(v_data);

        // Update the current length if this position extends it.
        if pos + 1 > self.len {
            self.len = pos + 1;
        }
    }

    /// Get a slice of the key cache for positions 0..seq_len.
    ///
    /// Returns a slice of length seq_len * n_kv_heads * head_dim.
    pub fn get_k(&self, layer: usize, seq_len: usize) -> &[f32] {
        let kv_dim = self.n_kv_heads * self.head_dim;
        &self.k[layer][..seq_len * kv_dim]
    }

    /// Get a slice of the value cache for positions 0..seq_len.
    ///
    /// Returns a slice of length seq_len * n_kv_heads * head_dim.
    pub fn get_v(&self, layer: usize, seq_len: usize) -> &[f32] {
        let kv_dim = self.n_kv_heads * self.head_dim;
        &self.v[layer][..seq_len * kv_dim]
    }

    /// Reset the cache, zeroing all data and setting length to 0.
    pub fn reset(&mut self) {
        for layer_k in &mut self.k {
            layer_k.fill(0.0);
        }
        for layer_v in &mut self.v {
            layer_v.fill(0.0);
        }
        self.len = 0;
    }
}
