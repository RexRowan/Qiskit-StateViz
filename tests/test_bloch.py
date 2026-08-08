import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector

from qiskit_stateviz import plot_bloch_multivector_interactive


def test_returns_plotly_figure():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)
    qc.ry(0.5, 2)
    sv = Statevector(qc)
    fig = plot_bloch_multivector_interactive(sv)
    assert isinstance(fig, go.Figure)


def test_accepts_density_matrix():
    qc = QuantumCircuit(2)
    qc.h(0)
    dm = DensityMatrix(qc)
    fig = plot_bloch_multivector_interactive(dm)
    assert isinstance(fig, go.Figure)


def test_one_scene_per_qubit():
    qc = QuantumCircuit(4)
    qc.h(range(4))
    sv = Statevector(qc)
    fig = plot_bloch_multivector_interactive(sv)
    layout_keys = [k for k in fig.layout if k.startswith("scene")]
    assert len(layout_keys) == 4


def test_entangled_qubit_has_near_zero_bloch_vector():
    # Bell state: each qubit's marginal is maximally mixed -> Bloch vector ~ 0.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sv = Statevector(qc)
    fig = plot_bloch_multivector_interactive(sv)

    # Tip markers are identified by their diamond symbol (see _vector_traces).
    tip_traces = [
        t for t in fig.data
        if isinstance(t, go.Scatter3d)
        and t.marker is not None
        and t.marker.symbol == "diamond"
    ]
    assert len(tip_traces) == 2
    for t in tip_traces:
        bx, by, bz = t.x[0], t.y[0], t.z[0]
        assert abs(bx) < 1e-8 and abs(by) < 1e-8 and abs(bz) < 1e-8


def test_nonzero_vector_has_populated_trail():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    sv = Statevector(qc)
    fig = plot_bloch_multivector_interactive(sv)

    tip_traces = [
        t for t in fig.data
        if isinstance(t, go.Scatter3d)
        and t.marker is not None
        and t.marker.symbol == "diamond"
    ]
    assert len(tip_traces) == 1
    tip = tip_traces[0]
    r = (tip.x[0] ** 2 + tip.y[0] ** 2 + tip.z[0] ** 2) ** 0.5
    assert r > 0.9  # pure single-qubit state -> Bloch vector on the surface


def test_default_vector_color_is_aqua():
    qc = QuantumCircuit(1)
    qc.h(0)
    sv = Statevector(qc)
    fig = plot_bloch_multivector_interactive(sv)
    tip_traces = [
        t for t in fig.data
        if isinstance(t, go.Scatter3d)
        and t.marker is not None
        and t.marker.symbol == "diamond"
    ]
    assert tip_traces[0].marker.color == "#00B4D8"


def test_dark_theme_applied():
    qc = QuantumCircuit(1)
    qc.h(0)
    sv = Statevector(qc)
    fig = plot_bloch_multivector_interactive(sv)
    assert fig.layout.template.layout.paper_bgcolor is not None or fig.layout.paper_bgcolor == "#111318"
    assert fig.layout.scene.bgcolor == "#111318"


def test_pole_labels_present():
    qc = QuantumCircuit(1)
    qc.h(0)
    sv = Statevector(qc)
    fig = plot_bloch_multivector_interactive(sv)
    text_traces = [t for t in fig.data if isinstance(t, go.Scatter3d) and t.mode == "text"]
    assert len(text_traces) == 1
    labels = set(text_traces[0].text)
    assert {"|0⟩", "|1⟩", "|+⟩", "|−⟩", "|+i⟩", "|−i⟩"} == labels


def test_pole_labels_have_distinct_textpositions():
    # Each of the six labels should get its own outward textposition, so
    # they spread apart instead of all rendering centered on their point
    # (the cause of the label-overlap bug).
    qc = QuantumCircuit(1)
    qc.h(0)
    sv = Statevector(qc)
    fig = plot_bloch_multivector_interactive(sv)
    text_trace = [t for t in fig.data if isinstance(t, go.Scatter3d) and t.mode == "text"][0]
    assert len(set(text_trace.textposition)) == 6


def test_scenes_use_free_orbit_rotation():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    sv = Statevector(qc)
    fig = plot_bloch_multivector_interactive(sv)
    assert fig.layout.scene.dragmode == "orbit"
    assert fig.layout.scene2.dragmode == "orbit"
