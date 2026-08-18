"""Cayley-graph emission spectrum and continuous-time quantum-walk dynamics.

A different kind of view from the rest of this package: instead of a single
qubit's ``Statevector``, this renders the eigenspectrum of a *walk*
Hamiltonian -- the adjacency matrix of a Cayley graph ``Cay(Z_n, S)`` -- as
an emission spectrum, alongside the continuous-time quantum walk (CTQW)
population dynamics that same Hamiltonian generates.

The actual diagonalization and Fourier-decomposition math lives in the
optional ``qiskit-eigenlight`` dependency; this module is a thin Plotly
rendering layer over it, kept in the same visual language as the rest of
qiskit-stateviz (dark theme, hover readouts, consistent color roles).
Install the optional dependency with::

    pip install qiskit-stateviz[spectrum]
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import StateVizError

try:
    from qiskit_eigenlight import (
        build_cayley_adjacency,
        girth as _girth,
        spectral_gap as _spectral_gap,
        spectral_lines as _spectral_lines,
        ctqw_populations as _ctqw_populations,
    )
    _HAS_EIGENLIGHT = True
except ImportError:  # pragma: no cover - exercised via test_spectrum guard test
    _HAS_EIGENLIGHT = False

_BG = "#111318"
_TEXT = "#e8e8ea"
_GRID_LIGHT = "#3a3f4a"
_GOLD = "#E7B84E"
_GREEN = "#7ED6A5"
_RED = "#E36A6A"

_SITE_PALETTE = [_GOLD, _GREEN, _RED, "#6FB1E0", "#B98BF0", "#63C7B2", "#E0A458", "#D9647C"]

# Emission-line color gradient, deliberately distinct from the site palette
# above: lines are colored by *frequency*, not by identity, so the top
# panel reads as an actual spectrum rather than a categorical chart.
_SPEC_STOPS = [
    (0.00, (109, 74, 255)),
    (0.25, (59, 130, 246)),
    (0.50, (34, 211, 238)),
    (0.70, (52, 211, 153)),
    (0.85, (251, 191, 36)),
    (1.00, (244, 63, 94)),
]


def _freq_to_color(t: float) -> str:
    t = min(1.0, max(0.0, t))
    for (t0, c0), (t1, c1) in zip(_SPEC_STOPS, _SPEC_STOPS[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0 or 1.0)
            r = round(c0[0] + (c1[0] - c0[0]) * f)
            g = round(c0[1] + (c1[1] - c0[1]) * f)
            b = round(c0[2] + (c1[2] - c0[2]) * f)
            return f"rgb({r},{g},{b})"
    return "rgb(244,63,94)"


def _require_eigenlight() -> None:
    if not _HAS_EIGENLIGHT:
        raise StateVizError(
            "plot_evolution_spectrum requires the optional 'qiskit-eigenlight' "
            "dependency, which isn't installed. Install with: "
            "pip install qiskit-stateviz[spectrum]"
        )


def plot_evolution_spectrum(
    n: int,
    generators: Iterable[int],
    *,
    start_vertex: int = 0,
    transition_operator: Optional[np.ndarray] = None,
    title: str = "Cayley graph emission spectrum",
    height: int = 640,
) -> go.Figure:
    """Render the emission spectrum and CTQW dynamics of ``Cay(Z_n, S)``.

    Builds the adjacency matrix of the circulant Cayley graph on the cyclic
    group ``Z_n`` with generating set ``S``, diagonalizes it, and shows two
    linked views of the same Hamiltonian:

    * **top panel** -- the Fourier spectrum of a probe observable's
      expectation value under free evolution: a spectral line at every
      eigenvalue gap ``|lambda_k - lambda_l|``, with height set by the true
      coherence amplitude ``|c_k c_l <k|T|l>|`` (see the
      ``qiskit-eigenlight`` README for exactly what ``T`` defaults to, and
      why that default is a stated simplification rather than a physical
      claim).
    * **bottom panel** -- the continuous-time quantum walk population
      ``|psi_i(t)|^2`` at each vertex, starting localized at
      ``start_vertex``. This is the same walk formalism behind
      ``qiskit-graph-walks``' ``WalkBasedLayout`` mixing signatures, just
      visualized directly instead of reduced to a routing heuristic.

    Args:
        n: group order for the cyclic group ``Z_n`` (``n >= 3``).
        generators: generating set S, an iterable of ints in ``[1, n-1]``.
            Symmetrized automatically -- you don't need to pass both ``s``
            and ``n - s``.
        start_vertex: vertex the walk starts localized at.
        transition_operator: optional ``(n, n)`` array overriding the
            default uniform all-pairs probe used for the emission
            spectrum.
        title: figure title (girth and spectral gap are appended live).
        height: figure height in pixels.

    Returns:
        A ``plotly.graph_objects.Figure`` with two stacked panels. The
        graph's girth and spectral gap are annotated directly in the title
        and also available on ``fig.layout.meta`` for programmatic use.

    Raises:
        StateVizError: if the optional ``qiskit-eigenlight`` dependency
            isn't installed, or if the graph parameters are invalid.
    """
    _require_eigenlight()

    try:
        A = build_cayley_adjacency(n, generators)
    except ValueError as e:
        # qiskit-eigenlight raises plain ValueError for invalid graph
        # parameters; re-raised here as StateVizError so every error a
        # caller sees from this package has the same type, regardless of
        # which layer it actually came from.
        raise StateVizError(str(e)) from e

    evals = np.linalg.eigh(A)[0]
    lines = _spectral_lines(A, start_vertex, transition_operator=transition_operator)
    times, pops = _ctqw_populations(A, start_vertex)

    g = _girth(A)
    gap = _spectral_gap(evals)
    gens_sorted = sorted({s % n for s in generators})

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Emission spectrum — Cay(Z{n}, {gens_sorted})",
            "Vertex occupation — |psi_i(t)|^2 (continuous-time quantum walk)",
        ),
        vertical_spacing=0.16,
    )

    max_freq = max((ln.freq for ln in lines), default=1e-6) or 1e-6
    for ln in lines:
        color = _freq_to_color(ln.freq / max_freq)
        h = 0.08 + ln.normalized * 0.92
        # layered, widened, low-opacity strokes underneath a sharp core
        # line -- Plotly has no native bloom filter, so glow is faked the
        # same way here as in the standalone qiskit-eigenlight renderer.
        for width, opacity in ((10, 0.10), (6, 0.16), (3, 0.24)):
            fig.add_trace(
                go.Scatter(
                    x=[ln.freq, ln.freq], y=[0, h],
                    mode="lines", line=dict(color=color, width=width),
                    opacity=opacity, hoverinfo="skip", showlegend=False,
                ),
                row=1, col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=[ln.freq, ln.freq], y=[0, h],
                mode="lines", line=dict(color=color, width=2.2),
                hoverinfo="text",
                hovertext=[f"{ln.k} → {ln.l}: ω = {ln.freq:.3f}, intensity = {ln.normalized:.2f}"] * 2,
                showlegend=False,
            ),
            row=1, col=1,
        )

    for i in range(pops.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=times, y=pops[:, i], mode="lines",
                line=dict(color=_SITE_PALETTE[i % len(_SITE_PALETTE)], width=2),
                name=f"vertex {i}",
                hovertemplate=f"vertex {i}<br>t=%{{x:.2f}}<br>pop=%{{y:.3f}}<extra></extra>",
            ),
            row=2, col=1,
        )

    gap_str = f"{gap:.3f}"
    girth_str = str(g) if np.isfinite(g) else "inf"
    fig.update_layout(
        title=f"{title} — girth {girth_str}, spectral gap {gap_str}",
        height=height,
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT),
        margin=dict(l=50, r=30, t=90, b=50),
        showlegend=pops.shape[1] <= 8,
        legend=dict(orientation="h", y=-0.16),
    )
    fig.update_xaxes(title_text="ω (energy units, ħ = 1)", gridcolor=_GRID_LIGHT, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="t", gridcolor=_GRID_LIGHT, row=2, col=1)
    fig.update_yaxes(title_text="population", gridcolor=_GRID_LIGHT, row=2, col=1)
    fig.update_annotations(font=dict(color=_TEXT))

    # go.Figure doesn't allow arbitrary attribute assignment (unlike
    # matplotlib's Figure) -- layout.meta is Plotly's supported slot for
    # exactly this kind of caller-facing metadata.
    fig.update_layout(meta={
        "n": n,
        "generators": gens_sorted,
        "start_vertex": start_vertex,
        "girth": g,
        "spectral_gap": gap,
    })
    return fig
