#!/usr/bin/env python3
import numpy as np
import cv2
import torch
from pynvml import *
import tensorflow as tf
import time

def benchmark_gpu_performance(
        matrix_size: int = 4096,
        warmup_iters: int = 3,
        benchmark_iters: int = 10,
        dtype: tf.dtypes.DType = tf.float32,
        device: str = "/GPU:0"
):
    """Accurate GPU matrix multiplication benchmark with proper synchronization."""

    # Validate GPU availability
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus and device.startswith("/GPU"):
        print("No GPU found - falling back to CPU", file=sys.stderr)
        device = "/CPU:0"

    # Configure GPU
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass  # Ignore if already configured

    print(f"\nBenchmarking {matrix_size}x{matrix_size} matrices on {device}...")

    with tf.device(device):
        # Create matrices
        a = tf.random.normal([matrix_size, matrix_size], dtype=dtype)
        b = tf.random.normal([matrix_size, matrix_size], dtype=dtype)

        # Warmup with sync
        print("Running warmup...")
        for _ in range(warmup_iters):
            c = tf.linalg.matmul(a, b)
            c.numpy()  # Force sync

        # Timed benchmark
        print(f"Running {benchmark_iters} iterations...")
        start_time = time.time()
        for _ in range(benchmark_iters):
            c = tf.linalg.matmul(a, b)
            c.numpy()  # Force sync after each op
        total_time = time.time() - start_time

        # Results
        avg_time_ms = (total_time / benchmark_iters) * 1000
        print("\nResults:")
        print(f"- Total time: {total_time:.4f} sec")
        print(f"- Avg per matmul: {avg_time_ms:.3f} ms")
        print(f"- Device: {tf.test.gpu_device_name() or 'CPU'}")
        print("\nDetailed GPU Info:")
        print("Matrix multiplication result:")
        print("NumPy:", np.__version__)
        print("TF:", tf.__version__)
        print("GPU:", tf.config.list_physical_devices('GPU'))
        print("cv2:", cv2.__version__)
        print(f'PyTorch: {torch.__version__}')
        print(f'PyTorch CUDA: {torch.cuda.is_available()}')
        print(f"CUDA Version: {tf.sysconfig.get_build_info()['cuda_version']}")
        print(f"CuDNN Version: {tf.sysconfig.get_build_info()['cudnn_version']}")

if __name__ == "__main__":
    benchmark_gpu_performance()