import torch
import torch.nn as nn
import torch.nn.functional as F

class CacheOptimizer:
    """
    Cache optimization utility to enhance model performance during inference
    and training by leveraging caching mechanisms and reusing computation.
    """

    def __init__(self, model, cache_size=1024):
        """
        Initialize CacheOptimizer.
        
        Args:
            model (nn.Module): The transformer-based model to optimize.
            cache_size (int): Maximum size of the cache to store previous computations.
        """
        self.model = model
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def forward_with_cache(self, input_ids, attention_mask=None):
        """
        Forward pass with caching for repeated computations. 
        If a computation for a specific input is cached, it will be reused.
        
        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor, optional): Attention mask for the input.
        
        Returns:
            Tensor: Model output, either from cache or computed.
        """
        cache_key = self._generate_cache_key(input_ids)
        
        if cache_key in self.cache:
            # Cache hit
            self.cache_hits += 1
            output = self.cache[cache_key]
        else:
            # Cache miss, compute and store result
            self.cache_misses += 1
            output = self.model(input_ids, attention_mask)
            
            # Store in cache if cache is not full
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = output
            else:
                # Cache replacement policy: Remove the oldest cache entry
                self._evict_cache_entry()
                self.cache[cache_key] = output

        return output

    def _generate_cache_key(self, input_ids):
        """
        Generate a unique cache key for the given input.
        
        Args:
            input_ids (Tensor): Input token IDs.
        
        Returns:
            str: Unique string representing the cache key.
        """
        return hash(input_ids.cpu().numpy().tobytes())

    def _evict_cache_entry(self):
        """
        Evict the oldest entry from the cache (FIFO policy).
        """
        if self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

    def clear_cache(self):
        """
        Clear the entire cache manually.
        """
        self.cache.clear()

    def cache_stats(self):
        """
        Return cache hit and miss statistics.
        
        Returns:
            dict: Cache hit and miss statistics.
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) * 100 if total > 0 else 0
        miss_rate = (self.cache_misses / total) * 100 if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate
        }

class CacheOptimizedModel(nn.Module):
    """
    A wrapper model that integrates cache optimization with a given transformer model.
    """
    
    def __init__(self, model, cache_size=1024):
        """
        Initialize the CacheOptimizedModel.
        
        Args:
            model (nn.Module): The transformer-based model to optimize.
            cache_size (int): Maximum size of the cache.
        """
        super(CacheOptimizedModel, self).__init__()
        self.cache_optimizer = CacheOptimizer(model, cache_size)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass using cache optimization.
        
        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor, optional): Attention mask for the input.
        
        Returns:
            Tensor: Model output, optimized with caching.
        """
        return self.cache_optimizer.forward_with_cache(input_ids, attention_mask)

if __name__ == "__main__":
    # Example usage of Cache Optimized Model

    # Placeholder model (replace with actual transformer model)
    class ExampleTransformer(nn.Module):
        def __init__(self, hidden_size=768, num_labels=2):
            super(ExampleTransformer, self).__init__()
            self.hidden_size = hidden_size
            self.num_labels = num_labels
            self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(12)])
            self.output_layer = nn.Linear(hidden_size, num_labels)
        
        def forward(self, input_ids, attention_mask=None):
            x = input_ids.float()  # Placeholder for actual embedding lookup
            for layer in self.layers:
                x = layer(x)
            return self.output_layer(x)

    # Initialize model
    transformer_model = ExampleTransformer()
    cache_optimized_model = CacheOptimizedModel(transformer_model, cache_size=1024)

    # Example input data (replace with actual data)
    input_data = torch.randint(0, 1000, (1, 128))  # Simulating input token IDs

    # Perform inference with cache optimization
    output_1 = cache_optimized_model(input_data)
    output_2 = cache_optimized_model(input_data)  # Cached result should be used

    # Print cache stats
    stats = cache_optimized_model.cache_optimizer.cache_stats()
    print(f"Cache Hits: {stats['cache_hits']}, Cache Misses: {stats['cache_misses']}, Hit Rate: {stats['hit_rate']:.2f}%")
