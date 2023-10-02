import warnings
from contextlib import contextmanager
import torch

@contextmanager
def suppress_warnings(category):
    warnings.filterwarnings("ignore", category=category)
    yield
    warnings.filterwarnings("default", category=category)


def print_cuda_memory_usage():
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Converting bytes to GB.
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**2)  # Converting bytes to MB.
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**2)  # Converting bytes to MB.
    free_memory = total_memory - (reserved_memory / (1024)) # Converting MB to GB.

    print(f"Total GPU Memory: {total_memory:.2f} GB")
    print(f"Allocated Memory: {allocated_memory:.2f} MB")
    print(f"Free Memory: {free_memory:.2f} GB")


