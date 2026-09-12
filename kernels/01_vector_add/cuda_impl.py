"""CUDA variants share semantics; extension compilation is lazy and cached."""

from common.python.cuda_extension import load_extension


def naive(a, b):
    return load_extension("vector_add").naive(a, b)


def optimized(a, b):
    return load_extension("vector_add").optimized(a, b)
