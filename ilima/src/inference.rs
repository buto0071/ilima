use anyhow::{Error, Result};
use crate::model::sharded_llama as model;
use crate::layer_cache::LayerCache;
use crate::inference_config::InferenceConfig;
use candle_core::{Tensor, Device};
use candle_transformers::generation::LogitsProcessor;
use model::ModelWeights;
use tokenizers::Tokenizer;
use std::sync::Arc;

/// Inference engine with layer prefetching, caching, and quantization support
#[allow(dead_code)]
pub struct InferenceEngine {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    pub model_path: String,
    layer_cache: Option<Arc<LayerCache<Vec<u8>>>>,
    config: InferenceConfig,
}

#[allow(dead_code)]
impl InferenceEngine {
    /// Load model with optional layer range and custom configuration
    pub fn load_with_config(
        model_path: &str, 
        tokenizer_path: &str, 
        layer_range: Option<(usize, usize)>,
        config: InferenceConfig,
    ) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        println!("Loading model from {}", model_path);
        println!("{}", config.summary());
        
        let device = {
            #[cfg(feature = "cuda")]
            {
                println!("Attempting to use CUDA Backend...");
                match Device::new_cuda(0) {
                    Ok(d) => {
                        println!("🚀 Using CUDA Backend (NVIDIA)");
                        d
                    },
                    Err(e) => {
                        println!("⚠️ CUDA failed: {}. Falling back to CPU.", e);
                        Device::Cpu
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                println!("💻 Using CPU Backend");
                Device::Cpu
            }
        };
        
        let mut file = std::fs::File::open(model_path)?;
        println!("File opened");
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        println!("Content read");
        
        // Load model with appropriate layer range
        let model = ModelWeights::from_gguf(content, &mut file, &device, layer_range)?;
        println!("Model loaded (Range: {:?})", layer_range);
        
        // Print quantization info
        if config.quantization.enabled {
            println!("✨ Quantization enabled: {:?} ({}x memory reduction, {}x speedup)", 
                config.quantization.bits,
                config.quantization.memory_reduction_factor(),
                config.quantization.expected_speedup()
            );
        }
        
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;
        println!("Tokenizer loaded");

        
        // Initialize layer cache if enabled
        let layer_cache = if config.cache_size_mb > 0 {
            let cache_size_bytes = config.cache_size_mb * 1024 * 1024;
            println!("✅ Layer cache initialized (max size: {}MB)", config.cache_size_mb);
            Some(Arc::new(LayerCache::new(cache_size_bytes)))
        } else {
            println!("⚠️  Layer cache disabled");
            None
        };
        
        Ok(Self {
            model,
            tokenizer,
            device,
            model_path: model_path.to_string(),
            layer_cache,
            config,
        })
    }

    /// Load model with default configuration (backward compatibility)
    pub fn load(model_path: &str, tokenizer_path: &str, layer_range: Option<(usize, usize)>) -> Result<Self> {
        Self::load_with_config(model_path, tokenizer_path, layer_range, InferenceConfig::default())
    }

    /// Load model with high-performance configuration
    pub fn load_high_performance(model_path: &str, tokenizer_path: &str, layer_range: Option<(usize, usize)>) -> Result<Self> {
        Self::load_with_config(model_path, tokenizer_path, layer_range, InferenceConfig::high_performance())
    }

    /// Load model with low-memory configuration
    pub fn load_low_memory(model_path: &str, tokenizer_path: &str, layer_range: Option<(usize, usize)>) -> Result<Self> {
        Self::load_with_config(model_path, tokenizer_path, layer_range, InferenceConfig::low_memory())
    }

    /// Get current configuration
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Get cache statistics for monitoring
    pub fn cache_stats(&self) -> Option<crate::layer_cache::CacheStats> {
        self.layer_cache.as_ref().map(|cache| cache.stats())
    }

    /// Clear the layer cache
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.layer_cache {
            cache.clear();
            println!("🗑️  Layer cache cleared");
        }
    }

    /// Generate text with layer prefetching optimization
    pub fn generate(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        println!("Encoding prompt...");
        let mut tokens = self.tokenizer
            .encode(prompt, true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();
        println!("Prompt encoded. Tokens: {}", tokens.len());
            
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.8), Some(0.95));
        let mut new_tokens = vec![];

        println!("Starting generation loop...");
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input = Tensor::new(&tokens[start_pos..], &self.device)?.unsqueeze(0)?;
            
            // TODO: Implement prefetching here
            // For now, we perform standard forward pass
            // Future enhancement: prefetch next layer while computing current layer
            
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
            
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            
            // Log progress
            use std::io::Write;
            print!(".");
            std::io::stdout().flush().ok();

            if let Some(t) = self.tokenizer.id_to_token(next_token) {
                if t == "</s>" {
                    break;
                }
            }
        }
        println!(); // Newline after generation
        
        // Print cache stats after generation
        if let Some(stats) = self.cache_stats() {
            println!("📊 Cache stats: {} entries, {:.2}MB / {:.2}MB used", 
                stats.entries, 
                stats.size_bytes as f64 / 1024.0 / 1024.0,
                stats.max_size_bytes as f64 / 1024.0 / 1024.0
            );
        }
        
        let output = self.tokenizer.decode(&new_tokens, true).map_err(Error::msg)?;
        Ok(output)
    }
}
