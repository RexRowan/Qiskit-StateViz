"""Interactive Bloch-multivector visualization for n-qubit states."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    bloch_vector_from_rho1,
    num_qubits_from_dim,
    partial_trace_to_single_qubit,
    to_density_matrix_array,
)

# Dark theme palette, shared across all traces in this module.
_BG = "#111318"
_GRID_BOLD = "#9aa0aa"
_GRID_LIGHT = "#3a3f4a"
_TEXT = "#e8e8ea"
_AXIS_LINE = "#555c68"
_ORIGIN = "#8a8f99"

# Standard Bloch-sphere basis labels are attached to their poles directly
# inside _pole_label_trace(), where each also gets a distinct textposition.


def _sphere_wireframe_traces():
    """A wireframe sphere: equator, two primary meridians (bold), two
    secondary meridians and two latitude circles (light), for a shape
    that reads clearly as a sphere without turning into a tangled glob
    of overlapping circles.
    """
    traces = []
    t = np.linspace(0, 2 * np.pi, 60)

    def circle(x, y, z, color, width):
        return go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines", line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False,
        )

    # Primary great circles: equator + the two meridians aligned with the
    # x and y axes. These three alone define the sphere's silhouette from
    # any angle.
    traces.append(circle(np.cos(t), np.sin(t), np.zeros_like(t), _GRID_BOLD, 3))
    traces.append(circle(np.cos(t), np.zeros_like(t), np.sin(t), _GRID_BOLD, 3))
    traces.append(circle(np.zeros_like(t), np.cos(t), np.sin(t), _GRID_BOLD, 3))

    # Secondary meridians at 45/135 degrees, for extra 3D shape cues.
    for phi in (np.pi / 4, 3 * np.pi / 4):
        traces.append(
            circle(np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t), _GRID_LIGHT, 1.5)
        )

    # Two latitude circles above/below the equator.
    for z in (0.5, -0.5):
        r = np.sqrt(1 - z ** 2)
        traces.append(circle(r * np.cos(t), r * np.sin(t), np.full_like(t, z), _GRID_LIGHT, 1.5))

    return traces


def _axis_lines():
    lines = []
    for coords in (
        {"x": [-1, 1], "y": [0, 0], "z": [0, 0]},
        {"x": [0, 0], "y": [-1, 1], "z": [0, 0]},
        {"x": [0, 0], "y": [0, 0], "z": [-1, 1]},
    ):
        lines.append(
            go.Scatter3d(
                x=coords["x"], y=coords["y"], z=coords["z"],
                mode="lines",
                line=dict(color=_AXIS_LINE, width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return lines


def _pole_label_trace():
    # Distinct textposition per pole so labels spread outward in different
    # projected directions rather than all being centered exactly on their
    # 3D point (which is what caused labels to crowd/overlap each other
    # and the vector at small subplot sizes).
    entries = [
        ((0, 0, 1), "|0⟩", "top center"),
        ((0, 0, -1), "|1⟩", "bottom center"),
        ((1, 0, 0), "|+⟩", "middle right"),
        ((-1, 0, 0), "|−⟩", "middle left"),
        ((0, 1, 0), "|+i⟩", "top right"),
        ((0, -1, 0), "|−i⟩", "bottom left"),
    ]
    xs = [x * 1.55 for (x, y, z), _, _ in entries]
    ys = [y * 1.55 for (x, y, z), _, _ in entries]
    zs = [z * 1.55 for (x, y, z), _, _ in entries]
    labels = [lbl for _, lbl, _ in entries]
    positions = [pos for _, _, pos in entries]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="text",
        text=labels,
        textfont=dict(size=13, color=_TEXT),
        textposition=positions,
        hoverinfo="skip",
        showlegend=False,
    )


def _vector_traces(bx: float, by: float, bz: float, q: int, color: str):
    """Render the Bloch vector as a chain of markers plus a tip marker.

    Plotly's WebGL 3D line width is unreliable across browsers/GPUs — a
    ``width=8`` line can silently render as a 1px hairline that gets lost
    against wireframe/axis clutter. A short trail of solid markers along
    the vector, plus a larger marker at the tip, is visible everywhere
    regardless of that limitation.
    """
    r = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    n_dots = max(2, int(np.ceil(r * 18)))
    ts = np.linspace(0, 1, n_dots)
    xs, ys, zs = bx * ts, by * ts, bz * ts

    trail = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=5, color=color, opacity=0.9),
        hoverinfo="skip",
        showlegend=False,
    )

    tip_hover = (
        f"qubit {q}<br>x={bx:.4f}<br>y={by:.4f}<br>z={bz:.4f}<br>"
        f"|r|={r:.4f}"
    )
    tip = go.Scatter3d(
        x=[bx], y=[by], z=[bz],
        mode="markers",
        marker=dict(size=10, color=color, symbol="diamond",
                    line=dict(color="#00272b", width=1)),
        hovertext=[tip_hover],
        hoverinfo="text",
        showlegend=False,
    )

    origin = go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=4, color=_ORIGIN),
        hovertext=[f"qubit {q} (origin)"],
        hoverinfo="text",
        showlegend=False,
    )

    return [origin, trail, tip]


def plot_bloch_multivector_interactive(
    state,
    *,
    vector_color: str = "#00B4D8",
    title: str = "Bloch multivector",
    height: int = 460,
) -> go.Figure:
    """Render one interactive Bloch sphere per qubit, side by side.

    Each sphere shows the reduced single-qubit Bloch vector, i.e. the
    expectation values of X, Y, Z on that qubit after tracing out all
    others. This matches ``qiskit.visualization.plot_bloch_multivector``
    semantically, but is fully interactive rather than a static
    matplotlib image: drag to freely orbit each sphere in any direction,
    scroll/pinch to zoom, and hover the vector tip for exact values. Each
    sphere is labeled at its six basis poles (|0⟩, |1⟩, |+⟩, |−⟩, |+i⟩,
    |−i⟩), each pushed outward with its own text anchor so the six labels
    don't crowd each other or the vector, and its numeric Bloch-vector
    coordinates are shown directly under the subplot title, so the values
    are visible without hovering. Rendered on a dark background.

    Note: as with Qiskit's own ``plot_bloch_multivector``, this view shows
    only single-qubit marginals. It cannot represent entanglement between
    qubits — a maximally entangled qubit's Bloch vector has zero length,
    even though the full joint state is far from mixed. Use
    :func:`qiskit_stateviz.qsphere.plot_qsphere_interactive` (pure states)
    to see multi-qubit structure directly.

    Args:
        state: a ``qiskit.quantum_info.Statevector`` or ``DensityMatrix``,
            or an equivalent array-like.
        vector_color: color of the Bloch vector trail/tip on each sphere.
            Defaults to an aqua blue (``#00B4D8``).
        title: overall figure title.
        height: figure height in pixels.

    Returns:
        A ``plotly.graph_objects.Figure`` with one freely-rotatable 3D
        subplot per qubit.
    """
    rho = to_density_matrix_array(state)
    n = num_qubits_from_dim(rho.shape[0])

    bloch_vectors = []
    for q in range(n):
        rho1 = partial_trace_to_single_qubit(rho, q, n)
        bloch_vectors.append(bloch_vector_from_rho1(rho1))

    specs = [[{"type": "scene"} for _ in range(n)]]
    subplot_titles = [
        f"qubit {q}<br>(x={bx:.3f}, y={by:.3f}, z={bz:.3f})"
        for q, (bx, by, bz) in enumerate(bloch_vectors)
    ]
    fig = make_subplots(
        rows=1, cols=n, specs=specs, subplot_titles=subplot_titles,
        horizontal_spacing=min(0.15, 0.6 / max(n, 1)),
    )

    for q, (bx, by, bz) in enumerate(bloch_vectors):
        for trace in _sphere_wireframe_traces():
            fig.add_trace(trace, row=1, col=q + 1)
        for line in _axis_lines():
            fig.add_trace(line, row=1, col=q + 1)
        fig.add_trace(_pole_label_trace(), row=1, col=q + 1)
        for trace in _vector_traces(bx, by, bz, q, vector_color):
            fig.add_trace(trace, row=1, col=q + 1)

    scene_updates = {}
    for q in range(n):
        key = "scene" if q == 0 else f"scene{q + 1}"
        scene_updates[key] = dict(
            xaxis=dict(visible=False, range=[-1.75, 1.75]),
            yaxis=dict(visible=False, range=[-1.75, 1.75]),
            zaxis=dict(visible=False, range=[-1.75, 1.75]),
            aspectmode="cube",
            dragmode="orbit",
            # Pulled back slightly (was 1.4) so poles/labels near the
            # edge of the sphere have breathing room instead of nearly
            # touching the subplot title above them.
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
            bgcolor=_BG,
        )

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=0, r=0, t=90, b=0),
        template="plotly_dark",
        paper_bgcolor=_BG,
        font=dict(color=_TEXT),
        **scene_updates,
    )
    fig.update_annotations(font=dict(color=_TEXT))
    return fig
