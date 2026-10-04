import numpy as np

from sneppx_kernels.kernels import dequantize_int8, gemm, quantize_int8


def test_gemm_matches_numpy():
    a = np.random.rand(4, 8)
    b = np.random.rand(8, 2)
    assert np.allclose(gemm(a, b), a @ b)


def test_quantize_dequantize_close():
    x = np.linspace(-1, 1, 32)
    q, s = quantize_int8(x)
    assert np.max(np.abs(dequantize_int8(q, s) - x)) < s
