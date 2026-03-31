"""
PolarQuant — Polar-coordinate vector quantization for embedding compression.

Implements the core ideas from Google Research's PolarQuant (AISTATS 2026):
1. Random orthogonal rotation to simplify data geometry
2. Cartesian-to-polar conversion via recursive pair-wise transforms
3. Scalar quantization of concentrated angles to 3-4 bits
4. Approximate inner-product reconstruction from quantized polar form

The key insight: after rotation, angle distributions become highly
concentrated around known values, so a fixed uniform grid quantizes
them with near-zero distortion — eliminating the per-block normalization
overhead that traditional quantizers carry.

Reference:
    https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

import structlog

logger = structlog.get_logger(__name__)


class PolarEncoding(NamedTuple):
    """Result of converting a batch of Cartesian vectors to polar form."""
    radii: NDArray[np.float32]       # (n,) — vector norms
    angles: NDArray[np.float32]      # (n, d-1) — polar angles in [0, pi] or [0, 2pi]


class QuantizedVectors(NamedTuple):
    """Compressed representation of a vector collection."""
    radii: NDArray[np.float16]             # (n,) — norms stored at half precision
    quantized_angles: NDArray[np.uint8]    # (n, d-1) — angles quantized to `bits` levels
    bits: int                              # quantization bit-width
    angle_min: NDArray[np.float32]         # (d-1,) — per-dimension minimum (for dequant)
    angle_max: NDArray[np.float32]         # (d-1,) — per-dimension maximum (for dequant)


def random_rotation_matrix(d: int, seed: int = 42) -> NDArray[np.float32]:
    """Generate a d x d random orthogonal matrix via QR decomposition.

    The rotation simplifies the geometry of arbitrary vector distributions,
    making angle distributions concentrated and amenable to uniform quantization.
    """
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(H)
    # Ensure a proper rotation (det = +1) by absorbing sign of R diagonal
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign[np.newaxis, :]
    return Q.astype(np.float32)


def apply_rotation(
    vectors: NDArray[np.float32],
    rotation: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply orthogonal rotation: v' = v @ R^T."""
    return vectors @ rotation.T


def cartesian_to_polar(vectors: NDArray[np.float32]) -> PolarEncoding:
    """Convert Cartesian vectors to hyperspherical (polar) coordinates.

    Uses the recursive pair-wise approach from PolarQuant:
    - Pair adjacent dimensions (x_{2i}, x_{2i+1}) -> (r_i, theta_i)
    - Recursively pair the resulting radii until a single final radius remains
    - Collect all angles at each recursion level

    For a d-dimensional vector this produces 1 radius and (d-1) angles.
    """
    n, d = vectors.shape
    all_angles: list[NDArray[np.float32]] = []

    current = vectors.copy()

    while current.shape[1] > 1:
        cols = current.shape[1]
        # If odd number of columns, carry the last one through as-is
        if cols % 2 == 1:
            carry = current[:, -1:]
            pairs = current[:, :-1]
        else:
            carry = None
            pairs = current

        num_pairs = pairs.shape[1] // 2
        x_even = pairs[:, 0::2]  # (n, num_pairs)
        x_odd = pairs[:, 1::2]   # (n, num_pairs)

        r = np.sqrt(x_even**2 + x_odd**2 + 1e-12)
        theta = np.arctan2(x_odd, x_even + 1e-12)  # in [-pi, pi]

        all_angles.append(theta)

        if carry is not None:
            current = np.hstack([r, carry])
        else:
            current = r

    # current is now (n, 1) — the final radius
    radii = current[:, 0]

    if all_angles:
        angles = np.hstack(all_angles)
    else:
        angles = np.empty((n, 0), dtype=np.float32)

    return PolarEncoding(
        radii=radii.astype(np.float32),
        angles=angles.astype(np.float32),
    )


def polar_to_cartesian(polar: PolarEncoding, d: int) -> NDArray[np.float32]:
    """Reconstruct Cartesian vectors from polar encoding (inverse transform).

    Reverses the recursive pair-wise polar conversion by replaying the
    recursion levels in reverse order.
    """
    n = polar.radii.shape[0]
    num_angles = polar.angles.shape[1] if polar.angles.ndim == 2 else 0

    # Determine the recursion structure: at each level, how many pairs and
    # whether there was a carry column.
    levels: list[tuple[int, bool]] = []
    cols = d
    angle_offset = 0
    offsets: list[int] = []
    while cols > 1:
        has_carry = cols % 2 == 1
        num_pairs = (cols - (1 if has_carry else 0)) // 2
        offsets.append(angle_offset)
        levels.append((num_pairs, has_carry))
        angle_offset += num_pairs
        cols = num_pairs + (1 if has_carry else 0)

    current = polar.radii.reshape(n, 1).astype(np.float32)

    for level_idx in range(len(levels) - 1, -1, -1):
        num_pairs, has_carry = levels[level_idx]
        off = offsets[level_idx]

        if has_carry:
            r_vals = current[:, :num_pairs]
            carry = current[:, num_pairs:]
        else:
            r_vals = current
            carry = None

        theta = polar.angles[:, off:off + num_pairs]

        x_even = r_vals * np.cos(theta)
        x_odd = r_vals * np.sin(theta)

        # Interleave: [x_even_0, x_odd_0, x_even_1, x_odd_1, ...]
        interleaved = np.empty((n, num_pairs * 2), dtype=np.float32)
        interleaved[:, 0::2] = x_even
        interleaved[:, 1::2] = x_odd

        if carry is not None:
            current = np.hstack([interleaved, carry])
        else:
            current = interleaved

    return current


def quantize_angles(
    angles: NDArray[np.float32],
    bits: int = 3,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]:
    """Uniform scalar quantization of angle values.

    After rotation, angles are concentrated in narrow bands, so uniform
    quantization with 2^bits levels achieves near-optimal distortion.

    Returns:
        (quantized, angle_min, angle_max) — quantized codes and the
        per-dimension min/max needed for dequantization.
    """
    num_levels = (1 << bits) - 1  # e.g. 7 for 3-bit

    angle_min = angles.min(axis=0)
    angle_max = angles.max(axis=0)
    span = angle_max - angle_min
    span[span < 1e-8] = 1e-8  # avoid division by zero

    normalized = (angles - angle_min) / span  # [0, 1]
    quantized = np.clip(np.round(normalized * num_levels), 0, num_levels)
    return quantized.astype(np.uint8), angle_min, angle_max


def dequantize_angles(
    quantized: NDArray[np.uint8],
    angle_min: NDArray[np.float32],
    angle_max: NDArray[np.float32],
    bits: int = 3,
) -> NDArray[np.float32]:
    """Reconstruct approximate angle values from quantized codes."""
    num_levels = (1 << bits) - 1
    span = angle_max - angle_min
    return angle_min + (quantized.astype(np.float32) / num_levels) * span


def encode_vectors(
    vectors: NDArray[np.float32],
    rotation: NDArray[np.float32],
    bits: int = 3,
) -> QuantizedVectors:
    """Full PolarQuant encode pipeline: rotate -> polar -> quantize."""
    rotated = apply_rotation(vectors, rotation)
    polar = cartesian_to_polar(rotated)
    q_angles, a_min, a_max = quantize_angles(polar.angles, bits=bits)

    return QuantizedVectors(
        radii=polar.radii.astype(np.float16),
        quantized_angles=q_angles,
        bits=bits,
        angle_min=a_min,
        angle_max=a_max,
    )


def decode_vectors(
    qv: QuantizedVectors,
    rotation: NDArray[np.float32],
    d: int,
) -> NDArray[np.float32]:
    """Reconstruct approximate Cartesian vectors from quantized polar form."""
    angles = dequantize_angles(qv.quantized_angles, qv.angle_min, qv.angle_max, qv.bits)
    polar = PolarEncoding(radii=qv.radii.astype(np.float32), angles=angles)
    rotated = polar_to_cartesian(polar, d)
    # Undo rotation: v = v' @ R  (since R is orthogonal, R^{-1} = R^T, so v' @ R)
    return rotated @ rotation


def approximate_inner_product(
    query: NDArray[np.float32],
    qv: QuantizedVectors,
    rotation: NDArray[np.float32],
    d: int,
) -> NDArray[np.float32]:
    """Compute approximate inner products between a query and quantized database vectors.

    Reconstructs database vectors from their quantized polar form and
    computes the dot product against the (full-precision) query.

    Args:
        query: (1, d) or (d,) query vector in original Cartesian space.
        qv: Quantized database vectors.
        rotation: The rotation matrix used during encoding.
        d: Original dimensionality.

    Returns:
        (n,) array of approximate inner product scores.
    """
    if query.ndim == 1:
        query = query.reshape(1, -1)

    db_approx = decode_vectors(qv, rotation, d)
    scores = (query @ db_approx.T).ravel()
    return scores


def save_polar_quant(
    path: Path,
    rotation: NDArray[np.float32],
    qv: QuantizedVectors,
    d: int,
) -> None:
    """Persist PolarQuant data to a single .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rotation=rotation,
        radii=np.asarray(qv.radii),
        quantized_angles=np.asarray(qv.quantized_angles),
        angle_min=np.asarray(qv.angle_min),
        angle_max=np.asarray(qv.angle_max),
        bits=np.array([qv.bits]),
        dim=np.array([d]),
    )
    logger.debug("polar_quant_saved", path=str(path))


def load_polar_quant(
    path: Path,
) -> tuple[NDArray[np.float32], QuantizedVectors, int]:
    """Load PolarQuant data from a .npz file.

    Returns:
        (rotation_matrix, quantized_vectors, original_dim)
    """
    path = Path(path)
    data = np.load(path)
    rotation = data["rotation"].astype(np.float32)
    qv = QuantizedVectors(
        radii=data["radii"].astype(np.float16),
        quantized_angles=data["quantized_angles"].astype(np.uint8),
        bits=int(data["bits"][0]),
        angle_min=data["angle_min"].astype(np.float32),
        angle_max=data["angle_max"].astype(np.float32),
    )
    d = int(data["dim"][0])
    logger.debug("polar_quant_loaded", path=str(path), vectors=qv.radii.shape[0])
    return rotation, qv, d
