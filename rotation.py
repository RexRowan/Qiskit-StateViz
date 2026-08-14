"""Animated single-qubit rotation: the Bloch sphere vs. the SU(2) state
that actually lives in Hilbert space.

A physical rotation of a spin-1/2 state by angle theta is implemented by
``R(theta) = exp(-i * theta * (n.sigma) / 2)``. Because n.sigma has
eigenvalues +-1, ``R(2*pi) = -I`` for *any* axis n and *any* initial
state: one full physical turn multiplies the state by an overall minus
sign that no measurement, and therefore no point on the Bloch sphere,
can ever detect. Only at ``theta = 4*pi`` does ``R`` return to +I.

This module renders that gap directly: one 3D Bloch sphere (period
360 degrees) side by side with a 2D dial tracking the SU(2) rotation
parameter theta/2 (period 720 degrees), animated together over a single
Plotly slider from 0 to 720 degrees.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .bloch import _axis_lines, _pole_label_trace, _sphere_wireframe_traces, _vector_traces
from .utils import PAULI_X, PAULI_Y, PAULI_Z, StateVizError, bloch_vector_from_rho1, to_statevector_array

_BG = "#111318"
_TEXT = "#e8e8ea"
_GRID_LIGHT = "#3a3f4a"
_GOLD = "#E7B84E"
_GREEN = "#7ED6A5"
_RED = "#E36A6A"

_PAULIS = {"x": PAULI_X, "y": PAULI_Y, "z": PAULI_Z}


def _axis_vector(axis) -> np.ndarray:
    """Coerce ``axis`` (one of 'x'/'y'/'z' or a 3-vector) into a unit numpy vector."""
    if isinstance(axis, str):
        key = axis.lower()
        if key not in ("x", "y", "z"):
            raise StateVizError(f"Unknown axis label {axis!r}; use 'x', 'y', 'z', or a 3-vector.")
        return np.eye(3)["xyz".index(key)]
    v = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise StateVizError("Rotation axis must be nonzero.")
    return v / norm


def _default_axis_orthogonal_to(bloch_vec: np.ndarray) -> np.ndarray:
    """Pick a unit axis perpendicular to ``bloch_vec`` so the SU(2) dial's
    overlap readout stays a clean real-valued oscillation (see module
    docstring / README note in the function below for why this matters).
    """
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, bloch_vec)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    v = seed - np.dot(seed, bloch_vec) * bloch_vec
    return v / np.linalg.norm(v)


def _su2_rotation(theta: float, n_hat: np.ndarray) -> np.ndarray:
    """R(theta) = exp(-i theta (n.sigma) / 2), a 2x2 unitary."""
    n_sigma = n_hat[0] * PAULI_X + n_hat[1] * PAULI_Y + n_hat[2] * PAULI_Z
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * n_sigma


def _dial_static_traces(radius: float = 1.0):
    t = np.linspace(0, 2 * np.pi, 80)
    circle = go.Scatter(
        x=radius * np.cos(t), y=radius * np.sin(t),
        mode="lines", line=dict(color=_GRID_LIGHT, width=2),
        hoverinfo="skip", showlegend=False,
    )
    return [circle]


def plot_spin_rotation_interactive(
    state=None,
    *,
    axis=None,
    n_frames: int = 121,
    vector_color: str = "#00B4D8",
    title: str = "Spin rotation: Bloch sphere vs. SU(2) state",
    height: int = 480,
) -> go.Figure:
    """Animate a spin-1/2 rotation from 0 to 720 degrees, side by side in
    physical space (the Bloch sphere) and in the SU(2) parameter that
    actually governs the Hilbert-space state.

    ``R(theta) = exp(-i theta (n.sigma) / 2)`` satisfies ``R(2*pi) = -I``
    and ``R(4*pi) = +I`` for *any* axis ``n`` and *any* initial state —
    a full physical turn always multiplies the state by an invisible
    global phase of -1. The left panel is the Bloch vector, which has
    period 360 degrees because it can't see that phase. The right panel
    is a dial at angle theta/2, the SU(2) rotation's own parameter, which
    only closes back up at 720 degrees. Drag the slider or press play to
    watch the two panels fall out of sync at 360 degrees and resync at
    720.

    Args:
        state: a single-qubit ``qiskit.quantum_info.Statevector`` (or
            equivalent array-like of length 2). Defaults to the |+i>
            state (Bloch vector (0, 1, 0)) used as the canonical example.
        axis: rotation axis, either 'x' / 'y' / 'z' or a 3-vector. If not
            given, an axis orthogonal to the initial state's Bloch vector
            is chosen automatically, which keeps the theta/2 dial's
            overlap readout a clean real-valued oscillation between -1
            and +1 rather than a generic complex trajectory.
        n_frames: number of animation frames spanning 0-720 degrees.
        vector_color: color of the Bloch vector.
        title: figure title (current angle/readout is appended live).
        height: figure height in pixels.

    Returns:
        A ``plotly.graph_objects.Figure`` with Play/Pause buttons and a
        slider running from 0 to 720 degrees.
    """
    if state is None:
        psi0 = np.array([1, 1j], dtype=complex) / np.sqrt(2)  # |+i>, Bloch (0,1,0)
    else:
        psi0 = to_statevector_array(state)
        if psi0.shape[0] != 2:
            raise StateVizError(
                "plot_spin_rotation_interactive only supports single-qubit states."
            )

    rho0 = np.outer(psi0, psi0.conj())
    b0 = np.array(bloch_vector_from_rho1(rho0))

    n_hat = _default_axis_orthogonal_to(b0) if axis is None else _axis_vector(axis)

    thetas = np.linspace(0, 4 * np.pi, n_frames)

    specs = [[{"type": "scene"}, {"type": "xy"}]]
    fig = make_subplots(
        rows=1, cols=2, specs=specs,
        subplot_titles=("Physical space — Bloch vector", "SU(2) parameter — theta/2 dial"),
        horizontal_spacing=0.12,
    )

    for trace in _sphere_wireframe_traces():
        fig.add_trace(trace, row=1, col=1)
    for line in _axis_lines():
        fig.add_trace(line, row=1, col=1)
    fig.add_trace(_pole_label_trace(), row=1, col=1)
    vector_traces = _vector_traces(*b0, 0, vector_color)
    for trace in vector_traces:
        fig.add_trace(trace, row=1, col=1)
    vector_trace_indices = list(range(len(fig.data) - len(vector_traces), len(fig.data)))

    for trace in _dial_static_traces():
        fig.add_trace(trace, row=1, col=2)
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 0], mode="lines", line=dict(color=_GOLD, width=3),
                   hoverinfo="skip", showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=[1], y=[0], mode="markers", marker=dict(size=12, color=_GOLD),
                   hoverinfo="text", hovertext=["theta/2 = 0.0 rad"], showlegend=False),
        row=1, col=2,
    )
    dial_trace_indices = [len(fig.data) - 2, len(fig.data) - 1]

    def frame_traces(theta):
        R = _su2_rotation(theta, n_hat)
        psi = R @ psi0
        rho = np.outer(psi, psi.conj())
        bx, by, bz = bloch_vector_from_rho1(rho)
        vtraces = _vector_traces(bx, by, bz, 0, vector_color)

        overlap = np.vdot(psi0, psi)
        half = theta / 2
        px, py = np.cos(half), np.sin(half)
        if np.cos(half) > 0.02:
            col = _GREEN
        elif np.cos(half) < -0.02:
            col = _RED
        else:
            col = _GOLD
        radius_line = go.Scatter(x=[0, px], y=[0, py], line=dict(color=col, width=3))
        dot = go.Scatter(
            x=[px], y=[py], marker=dict(color=col),
            hovertext=[f"theta/2 = {half:.3f} rad, overlap<psi0|psi> = {overlap.real:.3f}{overlap.imag:+.3f}i"],
        )
        return vtraces + [radius_line, dot], overlap

    frames = []
    for i, theta in enumerate(thetas):
        traces, overlap = frame_traces(theta)
        deg = np.degrees(theta)
        if abs(overlap - 1) < 0.01:
            verdict = "both back to start"
        elif abs(overlap + 1) < 0.01:
            verdict = "Bloch vector unchanged, state picked up a global minus sign"
        else:
            verdict = "mid-rotation"
        frames.append(
            go.Frame(
                data=traces,
                traces=vector_trace_indices + dial_trace_indices,
                name=f"{deg:.0f}",
                layout=dict(title=f"{title} — {deg:.0f} deg ({verdict})"),
            )
        )
    fig.frames = frames

    fig.update_layout(
        title=f"{title} — 0 deg",
        height=height,
        margin=dict(l=0, r=0, t=90, b=60),
        template="plotly_dark",
        paper_bgcolor=_BG,
        font=dict(color=_TEXT),
        scene=dict(
            xaxis=dict(visible=False, range=[-1.75, 1.75]),
            yaxis=dict(visible=False, range=[-1.75, 1.75]),
            zaxis=dict(visible=False, range=[-1.75, 1.75]),
            aspectmode="cube",
            dragmode="orbit",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
            bgcolor=_BG,
        ),
        xaxis2=dict(visible=False, range=[-1.3, 1.3], scaleanchor="y2"),
        yaxis2=dict(visible=False, range=[-1.3, 1.3]),
        updatemenus=[
            dict(
                type="buttons", showactive=False, x=0.02, y=-0.06, xanchor="left",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=35, redraw=True),
                                           fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode="immediate", transition=dict(duration=0))]),
                ],
            )
        ],
        sliders=[
            dict(
                active=0, x=0.02, y=-0.02, len=0.96,
                currentvalue=dict(prefix="theta = ", suffix=" deg", font=dict(color=_TEXT)),
                steps=[
                    dict(method="animate", label=f.name,
                         args=[[f.name], dict(mode="immediate",
                                               frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0))])
                    for f in fig.frames
                ],
            )
        ],
    )
    fig.update_annotations(font=dict(color=_TEXT))
    fig.add_annotation(
        text="+1", x=1.15, y=0, xref="x2", yref="y2", showarrow=False, font=dict(color=_GREEN, size=12),
    )
    fig.add_annotation(
        text="-1", x=-1.15, y=0, xref="x2", yref="y2", showarrow=False, font=dict(color=_RED, size=12),
    )
    return fig
