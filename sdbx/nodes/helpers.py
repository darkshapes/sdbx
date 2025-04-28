import gc
import io
import os
import re
import json
import base64

from functools import cache as function_cache, wraps

from sdbx.config import config

### CACHING ###


def generator_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If the cache exists and matches the arguments, return an iterator over the cached results
        if wrapper.cache and wrapper.cache_args == (args, kwargs):
            return iter(wrapper.cache)

        # If no cache exists, or arguments differ, create a new generator
        wrapper.cache = []
        wrapper.cache_args = (args, kwargs)

        def generator_with_cache():
            for result in func(*args, **kwargs):
                wrapper.cache.append(result)  # Cache the result as it's generated
                yield result  # Yield the result to the caller

        return generator_with_cache()

    # Initialize cache and arguments
    wrapper.cache = None
    wrapper.cache_args = None
    return wrapper


cache = lambda node: generator_cache(node) if node.generator else function_cache(node)

### NODE INFO NAMING ###


def rename_class(base, name):
    # Create a new class dynamically, inheriting from base_class
    new = type(name, (base,), {})

    # Set the __name__ and __qualname__ attributes to reflect the new name
    new.__name__ = name
    new.__qualname__ = name

    return new


def format_name(name):
    return " ".join(word[0].upper() + word[1:] if word else "" for word in re.split(r"_", name))


### NODE INFO TIMING ###

from functools import wraps
from time import time


def timing(callback):
    def decorator(f):
        @wraps(f)
        def wrap(instance, *args, **kwargs):
            ts = time()
            result = f(instance, *args, **kwargs)
            te = time()
            elapsed_time = te - ts
            # Use the class attribute 'name' for timing log
            print(f"Class: {instance.__class__.__name__} - Instance: {instance.name} - Elapsed Time: {elapsed_time:.4f} sec")
            callback(f"Class: {instance.__class__.__name__} - Instance: {instance.name} - Elapsed Time: {elapsed_time:.4f} sec")
            return result

        return wrap

    return decorator
