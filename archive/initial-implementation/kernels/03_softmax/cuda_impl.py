"""CUDA variants share semantics; extension compilation is lazy and cached."""

from common.python.cuda_extension import load_extension


def naive(x):
    return load_extension("softmax").naive(x)


def optimized(x):
    return load_extension("softmax").optimized(x)
