import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from qiskit_stateviz.utils import (
    StateVizError,
    bloch_vector_from_rho1,
    num_qubits_from_dim,
    partial_trace_to_single_qubit,
    to_density_matrix_array,
    to_statevector_array,
)


def test_to_statevector_array_from_statevector():
    sv = Statevector.from_label("01")
    arr = to_statevector_array(sv)
    assert arr.shape == (4,)
    assert np.isclose(np.linalg.norm(arr), 1.0)


def test_to_statevector_array_rejects_non_power_of_two():
    with pytest.raises(StateVizError):
        to_statevector_array(np.array([1, 0, 0], dtype=complex))


def test_to_statevector_array_rejects_unnormalized():
    with pytest.raises(StateVizError):
        to_statevector_array(np.array([1, 1], dtype=complex))


def test_to_density_matrix_array_promotes_pure_state():
    sv = Statevector.from_label("1")
    rho = to_density_matrix_array(sv)
    expected = np.array([[0, 0], [0, 1]], dtype=complex)
    assert np.allclose(rho, expected)


def test_partial_trace_matches_qiskit_reference():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.7, 2)
    sv = Statevector(qc)
    rho = to_density_matrix_array(sv)
    n = num_qubits_from_dim(rho.shape[0])

    for q in range(n):
        mine = partial_trace_to_single_qubit(rho, q, n)
        ref = partial_trace(DensityMatrix(sv), [i for i in range(n) if i != q]).data
        assert np.allclose(mine, ref, atol=1e-8), f"mismatch on qubit {q}"


def test_bloch_vector_matches_expectation_values():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.ry(1.1, 1)
    sv = Statevector(qc)
    rho = to_density_matrix_array(sv)
    n = num_qubits_from_dim(rho.shape[0])

    from qiskit.quantum_info import SparsePauliOp

    dm = DensityMatrix(sv)
    for q in range(n):
        rho1 = partial_trace_to_single_qubit(rho, q, n)
        computed = bloch_vector_from_rho1(rho1)

        def op(pauli, qubit=q):
            labels = ["I"] * n
            labels[n - 1 - qubit] = pauli
            return SparsePauliOp("".join(labels))

        reference = tuple(
            float(np.real(dm.expectation_value(op(p)))) for p in ("X", "Y", "Z")
        )
        assert np.allclose(computed, reference, atol=1e-8), f"mismatch on qubit {q}"


def test_bloch_vector_zero_for_maximally_mixed():
    rho1 = np.eye(2, dtype=complex) / 2
    x, y, z = bloch_vector_from_rho1(rho1)
    assert np.allclose([x, y, z], [0, 0, 0], atol=1e-10)


def test_bloch_vector_unit_for_computational_basis_zero():
    rho1 = np.array([[1, 0], [0, 0]], dtype=complex)
    x, y, z = bloch_vector_from_rho1(rho1)
    assert np.allclose([x, y, z], [0, 0, 1], atol=1e-10)
