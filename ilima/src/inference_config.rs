use serde::{Serialize, Deserialize};
use anyhow::Result;

/// Configuration for model quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization precision (4-bit or 8-bit)
    pub bits: QuantizationBits,
    /// Whether quantization is enabled
    pub enabled: bool,
    /// Block size for block-wise quantization
    pub block_size: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantizationBits {
    /// 4-bit quantization (maximum compression, slightly lower accuracy)
    Bits4,
    /// 8-bit quantization (balanced compression and accuracy)
    Bits8,
    /// No quantization (full precision)
    None,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            bits: QuantizationBits::Bits8,
            enabled: true,
            block_size: 32, // Standard block size for effective quantization
        }
    }
}

#[allow(dead_code)]
impl QuantizationConfig {
    /// Create a new quantization config with 4-bit precision
    pub fn bits4() -> Self {
        Self {
            bits: QuantizationBits::Bits4,
            enabled: true,
            block_size: 32,
        }
    }

    /// Create a new quantization config with 8-bit precision
    pub fn bits8() -> Self {
        Self {
            bits: QuantizationBits::Bits8,
            enabled: true,
            block_size: 32,
        }
    }

    /// Create a config with no quantization (full precision)
    pub fn none() -> Self {
        Self {
            bits: QuantizationBits::None,
            enabled: false,
            block_size: 0,
        }
    }

    /// Get expected memory reduction factor
    pub fn memory_reduction_factor(&self) -> f32 {
        if !self.enabled {
            return 1.0;
        }
        
        match self.bits {
            QuantizationBits::Bits4 => 8.0,  // 32-bit -> 4-bit = 8x reduction
            QuantizationBits::Bits8 => 4.0,  // 32-bit -> 8-bit = 4x reduction
            QuantizationBits::None => 1.0,
        }
    }

    /// Get expected speedup factor
    pub fn expected_speedup(&self) -> f32 {
        if !self.enabled {
            return 1.0;
        }
        
        match self.bits {
            QuantizationBits::Bits4 => 3.0,  // ~3x speedup based on AirLLM research
            QuantizationBits::Bits8 => 2.0,  // ~2x speedup
            QuantizationBits::None => 1.0,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.enabled && self.block_size == 0 {
            anyhow::bail!("Block size must be greater than 0 when quantization is enabled");
        }
        
        if self.enabled && self.block_size > 128 {
            anyhow::bail!("Block size should not exceed 128 for effective quantization");
        }
        
        Ok(())
    }
}

/// Configuration for inference optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Quantization settings
    pub quantization: QuantizationConfig,
    /// Enable layer prefetching
    pub enable_prefetch: bool,
    /// Layer cache size in MB (0 = disabled)
    pub cache_size_mb: usize,
    /// Maximum batch size for batched inference
    pub max_batch_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            quantization: QuantizationConfig::default(),
            enable_prefetch: true,
            cache_size_mb: 2048, // 2GB default
            max_batch_size: 1,   // Start with single-sample inference
        }
    }
}

impl InferenceConfig {
    /// Create a high-performance config (more memory, faster)
    pub fn high_performance() -> Self {
        Self {
            quantization: QuantizationConfig::bits4(),
            enable_prefetch: true,
            cache_size_mb: 4096, // 4GB cache
            max_batch_size: 4,
        }
    }

    /// Create a low-memory config (less memory, slower)
    pub fn low_memory() -> Self {
        Self {
            quantization: QuantizationConfig::bits4(),
            enable_prefetch: false,
            cache_size_mb: 512, // 512MB cache
            max_batch_size: 1,
        }
    }

    /// Create a balanced config
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Validate all settings
    pub fn validate(&self) -> Result<()> {
        self.quantization.validate()?;
        
        if self.max_batch_size == 0 {
            anyhow::bail!("Batch size must be at least 1");
        }
        
        if self.max_batch_size > 32 {
            anyhow::bail!("Batch size should not exceed 32 for memory efficiency");
        }
        
        Ok(())
    }

    /// Print configuration summary
    pub fn summary(&self) -> String {
        format!(
            "Inference Config:\n\
             - Quantization: {:?} ({}x memory reduction, {}x speedup)\n\
             - Prefetch: {}\n\
             - Cache: {}MB\n\
             - Max Batch: {}",
            self.quantization.bits,
            self.quantization.memory_reduction_factor(),
            self.quantization.expected_speedup(),
            if self.enable_prefetch { "enabled" } else { "disabled" },
            self.cache_size_mb,
            self.max_batch_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::bits4();
        assert_eq!(config.bits, QuantizationBits::Bits4);
        assert_eq!(config.memory_reduction_factor(), 8.0);
        assert_eq!(config.expected_speedup(), 3.0);
    }

    #[test]
    fn test_inference_config_validation() {
        let config = InferenceConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_presets() {
        let hp = InferenceConfig::high_performance();
        assert_eq!(hp.cache_size_mb, 4096);
        
        let lm = InferenceConfig::low_memory();
        assert_eq!(lm.cache_size_mb, 512);
    }
}
