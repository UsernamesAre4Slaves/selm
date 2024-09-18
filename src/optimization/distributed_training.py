"""
cache_optimization.py
---------------------
This module provides cache-aware optimizations for both training and inference,
which aim to improve data retrieval speed, minimize memory usage, and reduce
redundant computations.

The primary techniques employed include:
- Data caching for frequently used input batches
- Layer-wise caching for recurrent computation paths
- Result caching for frequently repeated inferences

"""

import functools
import logging
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheOptimization:
    def __init__(self, max_cache_size=100):
        """
        Initializes the cache optimization mechanism.

        :param max_cache_size: Maximum number of items to store in the cache.
        """
        self.max_cache_size = max_cache_size
        self.cache = defaultdict(dict)  # Store cached values, organized by task and input
        self.cache_order = []  # Keep track of the order in which items are added for LRU eviction
    
    def get_cache_key(self, inputs):
        """
        Generate a unique key for caching based on the model inputs.
        For simplicity, we use a hash of the inputs.

        :param inputs: Inputs to the model or a specific layer.
        :return: Unique key for cache lookup.
        """
        return hash(tuple(np.asarray(inputs).flatten()))

    def get_from_cache(self, task_name, inputs):
        """
        Retrieves a result from the cache if available.

        :param task_name: The task or model layer from which the result is expected.
        :param inputs: Inputs for which we want to retrieve the cached result.
        :return: Cached result if available, otherwise None.
        """
        cache_key = self.get_cache_key(inputs)
        return self.cache[task_name].get(cache_key, None)

    def save_to_cache(self, task_name, inputs, result):
        """
        Saves the computed result into the cache.

        :param task_name: The task or model layer generating the result.
        :param inputs: Inputs for which the result was computed.
        :param result: Result to be cached.
        """
        cache_key = self.get_cache_key(inputs)

        # If the cache is full, evict the least recently used item
        if len(self.cache_order) >= self.max_cache_size:
            oldest_task, oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_task][oldest_key]

        self.cache[task_name][cache_key] = result
        self.cache_order.append((task_name, cache_key))
        logger.info(f"Cached result for task: {task_name}")

    def clear_cache(self):
        """Clears all caches."""
        self.cache.clear()
        self.cache_order.clear()
        logger.info("Cache cleared.")

    def cache_decorator(self, task_name):
        """
        A decorator to automatically cache results of a model task or layer.
        
        :param task_name: Task or model layer to cache.
        :return: Decorated function with caching.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self.get_cache_key(args)

                # Check if result is already in the cache
                if cache_key in self.cache[task_name]:
                    logger.info(f"Retrieving from cache for task: {task_name}")
                    return self.cache[task_name][cache_key]

                # Otherwise, compute and cache the result
                result = func(*args, **kwargs)
                self.save_to_cache(task_name, args, result)
                return result
            return wrapper
        return decorator

# Example usage within model layers

class ModelLayer:
    def __init__(self, cache_optimization):
        """
        Example model layer that uses cache optimization.

        :param cache_optimization: Instance of CacheOptimization for result caching.
        """
        self.cache_optimization = cache_optimization

    @cache_optimization.cache_decorator(task_name="layer_forward_pass")
    def forward(self, inputs):
        """Dummy forward pass that benefits from caching."""
        logger.info("Performing forward pass computation.")
        result = np.dot(inputs, np.random.rand(inputs.shape[1], 10))  # Some computation
        return result

# Example of how to integrate cache optimization in training and inference
if __name__ == "__main__":
    cache_opt = CacheOptimization(max_cache_size=50)
    model_layer = ModelLayer(cache_optimization=cache_opt)

    # Example input data for inference
    example_input = np.random.rand(1, 100)

    # Perform inference, caching results
    result_1 = model_layer.forward(example_input)
    result_2 = model_layer.forward(example_input)  # This should hit the cache
    result_3 = model_layer.forward(np.random.rand(1, 100))  # New input, will not be cached

    # Clear the cache after a set of operations
    cache_opt.clear_cache()
