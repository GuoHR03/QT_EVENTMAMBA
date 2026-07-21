import numpy as np
import pytest

from backend.ellipse_decoder import decode_ellipse_vsa


def test_decode_ellipse_vsa_returns_five_finite_parameters():
    matrix = np.asarray(
        [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, 0.5, 0.5]],
        dtype=np.float32,
    )
    vector = np.asarray(
        [[0.9, 0.8, 0.85, 0.75, 0.1, 0.2, -0.1, 0.05]],
        dtype=np.float32,
    )

    decoded = decode_ellipse_vsa(vector, matrix)

    assert decoded.shape == (1, 5)
    assert np.isfinite(decoded).all()
    assert decoded[0, 2] >= decoded[0, 3]


def test_decode_ellipse_vsa_rejects_incompatible_matrix():
    with pytest.raises(ValueError, match="matrix_A"):
        decode_ellipse_vsa(
            np.ones((1, 8), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
        )
