import numpy as np
import plotly.graph_objects as go
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_stateviz import plot_qsphere_interactive


def test_returns_plotly_figure():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sv = Statevector(qc)
    fig = plot_qsphere_interactive(sv)
    assert isinstance(fig, go.Figure)


def test_accepts_raw_ndarray():
    sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    fig = plot_qsphere_interactive(sv)
    assert isinstance(fig, go.Figure)


def test_marker_count_matches_nonzero_amplitudes():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sv = Statevector(qc)  # Bell state: only |00> and |11> populated
    fig = plot_qsphere_interactive(sv, show_labels=False)

    marker_traces = [t for t in fig.data if isinstance(t, go.Scatter3d) and t.marker is not None and t.marker.size is not None]
    # last Scatter3d with (list-valued) marker sizing is the populated-amplitude trace
    points_trace = marker_traces[-1]
    assert len(points_trace.x) == 2


def test_prob_tol_above_all_populated_states_raises():
    # Equal superposition: each basis state has probability 0.5.
    # A prob_tol above that should exclude everything and raise.
    sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
    with pytest.raises(ValueError):
        plot_qsphere_interactive(sv, prob_tol=0.6)


def test_single_qubit_state():
    sv = Statevector.from_label("+")
    fig = plot_qsphere_interactive(sv)
    assert isinstance(fig, go.Figure)


def test_all_basis_states_labeled_not_just_populated():
    # Bell state: only |00> and |11> are populated, but |01> and |10>
    # should still appear as labeled "ghost" points showing the full basis.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sv = Statevector(qc)
    fig = plot_qsphere_interactive(sv, show_labels=True)

    all_labels = set()
    for t in fig.data:
        if isinstance(t, go.Scatter3d) and t.text:
            all_labels.update(t.text)

    assert all_labels == {"|00⟩", "|01⟩", "|10⟩", "|11⟩"}


def test_no_duplicate_labels_at_populated_poles():
    # Regression test: previously a separate pole-label trace and the
    # per-point label trace both drew "|000>"/"|111>" text at the same
    # coordinates whenever those states were populated (e.g. GHZ state).
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    sv = Statevector(qc)
    fig = plot_qsphere_interactive(sv, show_labels=True)

    label_counts = {}
    for t in fig.data:
        if isinstance(t, go.Scatter3d) and t.text:
            for lbl in t.text:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

    assert label_counts.get("|000⟩") == 1
    assert label_counts.get("|111⟩") == 1


def test_free_orbit_rotation_enabled():
    sv = Statevector.from_label("+")
    fig = plot_qsphere_interactive(sv)
    assert fig.layout.scene.dragmode == "orbit"


def test_dark_theme_applied():
    sv = Statevector.from_label("+")
    fig = plot_qsphere_interactive(sv)
    assert fig.layout.paper_bgcolor == "#111318"
    assert fig.layout.scene.bgcolor == "#111318"


def test_basis_state_labels_include_ket_notation():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sv = Statevector(qc)
    fig = plot_qsphere_interactive(sv, show_labels=True)
    points_trace = [
        t for t in fig.data
        if isinstance(t, go.Scatter3d) and t.marker is not None and t.marker.size is not None
    ][-1]
    assert set(points_trace.text) == {"|00⟩", "|11⟩"}
