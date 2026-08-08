"""Interactive Q-sphere visualization for n-qubit pure statevectors."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .utils import num_qubits_from_dim, to_statevector_array

# Dark theme palette, shared across all traces in this module.
_BG = "#111318"
_GRID_BOLD = "#9aa0aa"
_GRID_LIGHT = "#3a3f4a"
_TEXT = "#e8e8ea"
_GHOST_MARKER = "#5a5f6a"
_GHOST_TEXT = "#7d828c"
_STEM = "#7a8090"


def _hamming_weight(i: int) -> int:
    return bin(i).count("1")


def _ring_positions(indices: list[int], weight: int, n: int) -> list[tuple[float, float, float]]:
    """Place basis states of a given Hamming weight evenly around a latitude ring.

    weight=0 -> north pole, weight=n -> south pole, matching Qiskit's own
    plot_state_qsphere convention.
    """
    theta = np.pi * weight / n if n > 0 else 0.0
    z = np.cos(theta)
    r_xy = np.sin(theta)

    count = len(indices)
    positions = []
    for k in range(count):
        # Poles (weight 0 or n) have a single point; avoid divide-by-zero.
        phi = 2 * np.pi * k / count if count > 0 else 0.0
        x = r_xy * np.cos(phi)
        y = r_xy * np.sin(phi)
        positions.append((x, y, z))
    return positions


def _sphere_wireframe_traces(n: int):
    """A wireframe sphere: a latitude ring at each Hamming-weight level
    (the same rings basis states are placed on) plus several meridians,
    so the shape reads clearly as a sphere and the rings carry meaning
    (each one is exactly where a given weight's basis states sit).
    """
    traces = []
    t = np.linspace(0, 2 * np.pi, 60)

    def circle(x, y, z, color, width):
        return go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines", line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False,
        )

    bold, light = _GRID_BOLD, _GRID_LIGHT

    for w in range(n + 1):
        theta = np.pi * w / n if n > 0 else 0.0
        z = np.cos(theta)
        r = np.sin(theta)
        if r < 1e-9:
            continue
        color = bold if w in (0, n) or w == n / 2 else light
        traces.append(circle(r * np.cos(t), r * np.sin(t), np.full_like(t, z), color, 2))

    for phi in (0, np.pi / 2, np.pi / 4, 3 * np.pi / 4):
        color = bold if phi in (0, np.pi / 2) else light
        traces.append(circle(np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t), color, 2))

    return traces


def plot_qsphere_interactive(
    state,
    *,
    prob_tol: float = 1e-10,
    show_labels: bool = True,
    colorscale: str = "hsv",
    marker_scale: float = 40.0,
    title: str = "Q-sphere",
    height: int = 650,
) -> go.Figure:
    """Render an interactive Plotly Q-sphere for an n-qubit pure state.

    Args:
        state: a ``qiskit.quantum_info.Statevector``, or any array-like of
            length 2**n giving the complex amplitudes.
        prob_tol: basis states with probability below this threshold are
            omitted from the plot.
        show_labels: whether to label each basis state with its ket
            notation. Every basis state is labeled, not just populated
            ones — unpopulated states are shown as small faint "ghost"
            points so the full 2**n structure of the sphere stays visible.
        colorscale: a Plotly colorscale name used to map relative phase to
            color (default ``"hsv"``, appropriate for a cyclic quantity).
        marker_scale: scales marker size, which is proportional to
            probability.
        title: plot title.
        height: figure height in pixels.

    Returns:
        A ``plotly.graph_objects.Figure``. Call ``.show()`` to display it,
        or return it directly from a Jupyter cell. Drag freely (360-degree
        orbit) to rotate, scroll/pinch to zoom. Rendered on a dark
        background.
    """
    amplitudes = to_statevector_array(state)
    dim = amplitudes.shape[0]
    n = num_qubits_from_dim(dim)

    probs = np.abs(amplitudes) ** 2
    keep = np.where(probs > prob_tol)[0]
    if keep.size == 0:
        raise ValueError("All basis state probabilities are below prob_tol.")

    # Recenter phase relative to the first surviving amplitude, matching
    # Qiskit's convention of removing an unobservable global phase.
    ref_phase = np.angle(amplitudes[keep[0]])

    # Every possible basis state gets a ring position and a label — not
    # just the populated ones — so the full structure of the sphere is
    # visible. This also removes the earlier double-labeling bug, where
    # a separate "pole label" trace and the per-point label trace both
    # drew text at the |00...0> / |11...1> poles whenever those states
    # happened to be populated.
    all_indices = np.arange(dim)
    all_weights = np.array([_hamming_weight(int(i)) for i in all_indices])

    all_xs, all_ys, all_zs, all_idx_ordered = [], [], [], []
    for w in range(n + 1):
        idx_at_weight = all_indices[all_weights == w]
        positions = _ring_positions(list(idx_at_weight), w, n)
        for idx, (x, y, z) in zip(idx_at_weight, positions):
            all_xs.append(x)
            all_ys.append(y)
            all_zs.append(z)
            all_idx_ordered.append(int(idx))

    all_xs = np.array(all_xs)
    all_ys = np.array(all_ys)
    all_zs = np.array(all_zs)
    all_idx_ordered = np.array(all_idx_ordered)
    is_populated = np.isin(all_idx_ordered, keep)

    # textposition per point, chosen by hemisphere so labels are pushed
    # away from the sphere's center in projection rather than centered
    # exactly on the marker (which is what caused labels to visually
    # collide with the wireframe/other labels at small sizes).
    textpositions = np.where(all_zs >= 0, "top center", "bottom center")

    fig = go.Figure()

    for trace in _sphere_wireframe_traces(n):
        fig.add_trace(trace)

    # Stems from center to each *populated* amplitude only — stems for
    # unpopulated ghost points would just add clutter with no information.
    for i, x, y, z in zip(all_idx_ordered, all_xs, all_ys, all_zs):
        if i not in keep:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=[0, x], y=[0, y], z=[0, z],
                mode="lines",
                line=dict(color=_STEM, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Ghost markers: every basis state that currently has ~zero amplitude,
    # shown faint with a label, so the full basis is always visible.
    ghost_mask = ~is_populated
    if show_labels and ghost_mask.any():
        ghost_labels = [
            f"|{format(int(i), f'0{n}b')}⟩" for i in all_idx_ordered[ghost_mask]
        ]
        fig.add_trace(
            go.Scatter3d(
                x=all_xs[ghost_mask], y=all_ys[ghost_mask], z=all_zs[ghost_mask],
                mode="markers+text",
                marker=dict(size=4, color=_GHOST_MARKER),
                text=ghost_labels,
                textfont=dict(size=11, color=_GHOST_TEXT),
                textposition=list(textpositions[ghost_mask]),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    elif ghost_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=all_xs[ghost_mask], y=all_ys[ghost_mask], z=all_zs[ghost_mask],
                mode="markers",
                marker=dict(size=4, color=_GHOST_MARKER),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    pop_idx = all_idx_ordered[is_populated]
    pop_xs, pop_ys, pop_zs = all_xs[is_populated], all_ys[is_populated], all_zs[is_populated]
    pop_probs = probs[pop_idx]
    pop_phases = (np.angle(amplitudes[pop_idx]) - ref_phase + np.pi) % (2 * np.pi) - np.pi
    pop_sizes = marker_scale * np.sqrt(pop_probs / pop_probs.max())
    pop_labels = [f"|{format(int(i), f'0{n}b')}⟩" for i in pop_idx]

    hover_text = [
        f"{lbl}<br>amplitude: {amplitudes[i]:.4f}<br>"
        f"probability: {p:.4f}<br>phase: {ph:.4f} rad"
        for lbl, i, p, ph in zip(pop_labels, pop_idx, pop_probs, pop_phases)
    ]

    fig.add_trace(
        go.Scatter3d(
            x=pop_xs, y=pop_ys, z=pop_zs,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=pop_sizes,
                color=pop_phases,
                colorscale=colorscale,
                cmin=-np.pi,
                cmax=np.pi,
                colorbar=dict(title="phase (rad)", tickfont=dict(color=_TEXT), title_font=dict(color=_TEXT)),
                line=dict(color=_BG, width=0.5),
            ),
            text=pop_labels if show_labels else None,
            textfont=dict(size=13, color=_TEXT),
            textposition=list(textpositions[is_populated]) if show_labels else None,
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        paper_bgcolor=_BG,
        font=dict(color=_TEXT),
        scene=dict(
            xaxis=dict(visible=False, range=[-1.3, 1.3]),
            yaxis=dict(visible=False, range=[-1.3, 1.3]),
            zaxis=dict(visible=False, range=[-1.3, 1.3]),
            aspectmode="cube",
            dragmode="orbit",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
            bgcolor=_BG,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
