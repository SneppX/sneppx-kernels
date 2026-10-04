"""Reference kernels: GEMM and INT8 quantization (NumPy)."""

import numpy as np


def gemm(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a @ b


def quantize_int8(x, scale=None):
    x = np.asarray(x, dtype=np.float64)
    if scale is None:
        scale = np.max(np.abs(x)) / 127.0 if np.max(np.abs(x)) > 0 else 1.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    return np.asarray(q, dtype=np.float64) * scale
