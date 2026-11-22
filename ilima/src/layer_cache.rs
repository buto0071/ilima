use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use anyhow::Result;

/// A simple single-lock LRU cache for managing layer data without lock-order deadlock risk
#[allow(dead_code)]
pub struct LayerCache<T> {
    inner: Arc<Mutex<CacheInner<T>>>,
    max_size: usize,
}

#[allow(dead_code)]
struct CacheInner<T> {
    map: HashMap<String, CacheEntry<T>>,
    current_size: usize,
}

#[allow(dead_code)]
struct CacheEntry<T> {
    data: T,
    last_used: std::time::Instant,
    size_bytes: usize,
}

#[allow(dead_code)]
impl<T: Clone> LayerCache<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheInner {
                map: HashMap::new(),
                current_size: 0,
            })),
            max_size,
        }
    }

    /// Insert a new entry into the cache
    pub fn insert(&self, key: String, data: T, size_bytes: usize) -> Result<()> {
        let mut guard = self.inner.lock().unwrap();

        // Evict entries if we exceed max size
        while guard.current_size + size_bytes > self.max_size && !guard.map.is_empty() {
            Self::evict_lru(&mut guard);
        }

        // Insert new entry
        guard.map.insert(
            key,
            CacheEntry {
                data,
                last_used: std::time::Instant::now(),
                size_bytes,
            },
        );
        guard.current_size += size_bytes;

        Ok(())
    }

    /// Get an entry from the cache
    pub fn get(&self, key: &str) -> Option<T> {
        let mut guard = self.inner.lock().unwrap();
        if let Some(entry) = guard.map.get_mut(key) {
            entry.last_used = std::time::Instant::now();
            Some(entry.data.clone())
        } else {
            None
        }
    }

    /// Check if a key exists in the cache
    pub fn contains(&self, key: &str) -> bool {
        let guard = self.inner.lock().unwrap();
        guard.map.contains_key(key)
    }

    /// Evict the least recently used entry
    fn evict_lru(inner: &mut CacheInner<T>) {
        if let Some(key) = inner
            .map
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(k, _)| k.clone())
        {
            if let Some(entry) = inner.map.remove(&key) {
                inner.current_size = inner.current_size.saturating_sub(entry.size_bytes);
                println!("Evicted layer cache entry: {} (freed {} bytes)", key, entry.size_bytes);
            }
        }
    }

    /// Get current cache statistics
    pub fn stats(&self) -> CacheStats {
        let guard = self.inner.lock().unwrap();
        CacheStats {
            entries: guard.map.len(),
            size_bytes: guard.current_size,
            max_size_bytes: self.max_size,
        }
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.map.clear();
        guard.current_size = 0;
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub size_bytes: usize,
    pub max_size_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_get() {
        let cache = LayerCache::new(1000);
        cache.insert("key1".to_string(), vec![1, 2, 3], 100).unwrap();
        
        assert!(cache.contains("key1"));
        let data = cache.get("key1").unwrap();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = LayerCache::new(250); // Small cache
        
        cache.insert("key1".to_string(), vec![1], 100).unwrap();
        cache.insert("key2".to_string(), vec![2], 100).unwrap();
        
        // This should trigger eviction of key1
        cache.insert("key3".to_string(), vec![3], 100).unwrap();
        
        assert!(!cache.contains("key1"));
        assert!(cache.contains("key2"));
        assert!(cache.contains("key3"));
    }
}
