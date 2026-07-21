"""NumPy implementation of the ellipse VSA decoder used at inference time."""

import numpy as np


def decode_ellipse_vsa(predict_vector, matrix_a):
    vector = np.asarray(predict_vector, dtype=np.float32)
    matrix = np.asarray(matrix_a, dtype=np.float32)
    if vector.ndim == 1:
        vector = vector[None]
    if vector.ndim != 2 or vector.shape[1] % 2:
        raise ValueError(f"Ellipse VSA output must be [B, 2D], got {vector.shape}")
    dimension = vector.shape[1] // 2
    if matrix.shape != (2, dimension):
        raise ValueError(
            f"matrix_A must have shape (2, {dimension}), got {matrix.shape}"
        )

    real_part, imag_part = np.split(vector, 2, axis=1)
    magnitude = np.sqrt(real_part * real_part + imag_part * imag_part)
    phase = np.arctan2(imag_part, real_part)
    position = phase @ (np.float32(0.01) * matrix.T)

    clipped = np.clip(magnitude, np.float32(0.0), np.float32(1.0))
    z_value = np.float32(1.0) - clipped
    inverse_j0 = (
        np.float32(2.0)
        * np.sqrt(z_value)
        * (
            np.float32(1.0)
            + np.float32(0.125) * z_value
            + np.float32(13.0 / 384.0) * z_value * z_value
        )
        / np.float32(10.0)
    )

    x_coords = matrix[0]
    y_coords = matrix[1]
    phi = np.stack(
        (x_coords * x_coords, 2 * x_coords * y_coords, y_coords * y_coords),
        axis=1,
    )
    decoded = []
    for batch_index in range(vector.shape[0]):
        coefficients = np.linalg.lstsq(
            phi,
            (inverse_j0[batch_index] ** 2)[:, None],
            rcond=None,
        )[0].reshape(-1)
        ellipse_matrix = np.asarray(
            [
                [coefficients[0], coefficients[1]],
                [coefficients[1], coefficients[2]],
            ],
            dtype=np.float32,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(ellipse_matrix)
        semi_minor = np.sqrt(max(abs(float(eigenvalues[0])), 1e-8))
        semi_major = np.sqrt(max(abs(float(eigenvalues[1])), 1e-8))
        major_vector = eigenvectors[:, 1]
        if major_vector[0] < 0:
            major_vector = -major_vector
        angle = float(np.arctan2(major_vector[1], major_vector[0]))
        if abs(semi_major - semi_minor) < 1e-4:
            angle = 0.0
        if angle > np.pi / 2 - 1e-5:
            angle -= np.pi
        decoded.append(
            [
                float(position[batch_index, 0]),
                float(position[batch_index, 1]),
                semi_major,
                semi_minor,
                angle,
            ]
        )
    return np.asarray(decoded, dtype=np.float32)
