"""Run with python -m kernels.03_softmax.benchmark."""

from benchmarks.run_all import main

if __name__ == "__main__":
    main(default_kernel="softmax")
