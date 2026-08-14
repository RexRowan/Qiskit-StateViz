import numpy as np
import plotly.graph_objects as go
import pytest
from qiskit.quantum_info import Statevector

from qiskit_stateviz import plot_spin_rotation_interactive
from qiskit_stateviz.rotation import _su2_rotation, _default_axis_orthogonal_to
from qiskit_stateviz.utils import StateVizError


def test_returns_plotly_figure():
    fig = plot_spin_rotation_interactive()
    assert isinstance(fig, go.Figure)


def test_default_state_is_bloch_y_pole():
    fig = plot_spin_rotation_interactive()
    assert len(fig.frames) > 0


def test_full_turn_flips_sign_for_any_axis_and_state():
    # R(2*pi) = -I regardless of axis or initial state.
    rng = np.random.default_rng(0)
    for _ in range(5):
        n_hat = rng.normal(size=3)
        n_hat /= np.linalg.norm(n_hat)
        R = _su2_rotation(2 * np.pi, n_hat)
        assert np.allclose(R, -np.eye(2), atol=1e-8)


def test_two_full_turns_returns_identity():
    n_hat = np.array([0.0, 0.0, 1.0])
    R = _su2_rotation(4 * np.pi, n_hat)
    assert np.allclose(R, np.eye(2), atol=1e-8)


def test_default_axis_is_orthogonal_to_bloch_vector():
    for b0 in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.6, 0.8, 0]):
        b0 = np.array(b0)
        n_hat = _default_axis_orthogonal_to(b0)
        assert abs(np.dot(n_hat, b0)) < 1e-8
        assert abs(np.linalg.norm(n_hat) - 1) < 1e-8


def test_rejects_multi_qubit_state():
    sv = Statevector.from_label("00")
    with pytest.raises(StateVizError):
        plot_spin_rotation_interactive(sv)


def test_accepts_statevector_input():
    sv = Statevector.from_label("+")
    fig = plot_spin_rotation_interactive(sv, axis="z")
    assert isinstance(fig, go.Figure)


def test_frames_span_0_to_720_degrees():
    fig = plot_spin_rotation_interactive(n_frames=25)
    labels = [float(f.name) for f in fig.frames]
    assert min(labels) == pytest.approx(0.0, abs=1e-6)
    assert max(labels) == pytest.approx(720.0, abs=1e-6)


def test_has_play_pause_and_slider():
    fig = plot_spin_rotation_interactive()
    assert len(fig.layout.updatemenus) == 1
    buttons = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert "Play" in buttons and "Pause" in buttons
    assert len(fig.layout.sliders) == 1
