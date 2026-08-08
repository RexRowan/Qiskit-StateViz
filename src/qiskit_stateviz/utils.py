"""Shared helpers for coercing Qiskit state objects into plain numpy arrays."""

from __future__ import annotations

import numpy as np


class StateVizError(Exception):
    """Raised for invalid or unsupported input states."""


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def to_statevector_array(state) -> np.ndarray:
    """Coerce a Qiskit ``Statevector``, list, or ndarray into a 1D complex ndarray.

    Accepts:
      * ``qiskit.quantum_info.Statevector``
      * any object exposing a ``.data`` attribute that is array-like
      * a plain list / tuple / numpy array of amplitudes

    Raises:
        StateVizError: if the input is not a valid pure-state vector of
            dimension 2**n for some integer n, or is not normalized.
    """
    data = state.data if hasattr(state, "data") else state
    arr = np.asarray(data, dtype=complex).flatten()

    if not _is_power_of_two(arr.shape[0]):
        raise StateVizError(
            f"Statevector length {arr.shape[0]} is not a power of 2 "
            "(qiskit-stateviz only supports qubit systems)."
        )

    norm = np.linalg.norm(arr)
    if not np.isclose(norm, 1.0, atol=1e-6):
        raise StateVizError(
            f"Input state is not normalized (||psi|| = {norm:.6f}). "
            "Pass a valid Statevector or normalized amplitude array."
        )

    return arr


def to_density_matrix_array(state) -> np.ndarray:
    """Coerce a Qiskit ``DensityMatrix``, ``Statevector``, or ndarray into a
    2D density matrix ndarray.

    If given a pure statevector-like input, promotes it to |psi><psi|.
    """
    data = state.data if hasattr(state, "data") else state
    arr = np.asarray(data, dtype=complex)

    if arr.ndim == 1:
        sv = to_statevector_array(arr)
        return np.outer(sv, sv.conj())

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise StateVizError(
            f"Expected a square density matrix, got shape {arr.shape}."
        )

    if not _is_power_of_two(arr.shape[0]):
        raise StateVizError(
            f"Density matrix dimension {arr.shape[0]} is not a power of 2 "
            "(qiskit-stateviz only supports qubit systems)."
        )

    return arr


def num_qubits_from_dim(dim: int) -> int:
    return int(np.log2(dim))


def partial_trace_to_single_qubit(rho: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Partial trace of an n-qubit density matrix down to a single qubit's 2x2 rho.

    Uses Qiskit's little-endian qubit ordering convention (qubit 0 is the
    least-significant / rightmost bit in the computational basis label).

    Implemented via ``np.einsum`` with explicit index letters: this is O(1)
    to get right for arbitrary n, unlike manual axis-bookkeeping with
    repeated ``np.trace`` calls (which is easy to get subtly wrong).
    """
    import string

    rho_t = rho.reshape([2] * n + [2] * n)

    # Axis i (0..n-1) is the bra index for qubit (n-1-i); axis n+i is the
    # matching ket index for the same qubit. This is standard row-major
    # reshape order combined with Qiskit's little-endian labeling.
    #
    # The kept qubit needs *distinct* bra/ket letters (it's not traced);
    # every traced qubit shares one letter between its bra and ket axis.
    kept_bra, kept_ket = "y", "z"
    bra_letters = []
    ket_letters = []
    pool = [c for c in string.ascii_lowercase if c not in (kept_bra, kept_ket)]
    if n - 1 > len(pool):
        raise StateVizError(f"Too many qubits ({n}) for einsum-based partial trace.")
    trace_letter_iter = iter(pool)
    for i in range(n):
        this_qubit = n - 1 - i
        if this_qubit == qubit:
            bra_letters.append(kept_bra)
            ket_letters.append(kept_ket)
        else:
            L = next(trace_letter_iter)
            bra_letters.append(L)
            ket_letters.append(L)

    subscript = "".join(bra_letters) + "".join(ket_letters) + f"->{kept_bra}{kept_ket}"
    return np.einsum(subscript, rho_t)


PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def bloch_vector_from_rho1(rho1: np.ndarray) -> tuple[float, float, float]:
    """Compute the (x, y, z) Bloch vector components of a single-qubit density matrix."""
    x = np.real(np.trace(rho1 @ PAULI_X))
    y = np.real(np.trace(rho1 @ PAULI_Y))
    z = np.real(np.trace(rho1 @ PAULI_Z))
    return float(x), float(y), float(z)
