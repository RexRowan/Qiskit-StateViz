import numpy as np
import plotly.graph_objects as go
import pytest

from qiskit_stateviz import plot_evolution_spectrum
from qiskit_stateviz.spectrum import _freq_to_color, _require_eigenlight
from qiskit_stateviz.utils import StateVizError


def test_returns_plotly_figure():
    fig = plot_evolution_spectrum(8, {1, 2})
    assert isinstance(fig, go.Figure)


def test_has_two_subplot_titles():
    fig = plot_evolution_spectrum(8, {1, 2})
    titles = [a.text for a in fig.layout.annotations if a.text]
    assert any("Emission spectrum" in t for t in titles)
    assert any("Vertex occupation" in t for t in titles)


def test_eigenlight_meta_matches_direct_computation():
    from qiskit_eigenlight import build_cayley_adjacency, girth, spectral_gap

    fig = plot_evolution_spectrum(12, {1, 5}, start_vertex=0)
    A = build_cayley_adjacency(12, {1, 5})
    evals = np.linalg.eigh(A)[0]

    meta = fig.layout.meta
    assert meta["n"] == 12
    assert meta["generators"] == [1, 5]
    assert meta["girth"] == girth(A)
    assert meta["spectral_gap"] == pytest.approx(spectral_gap(evals))


def test_generators_normalized_and_sorted_in_meta():
    fig = plot_evolution_spectrum(8, {9, 3})  # 9 mod 8 == 1
    assert fig.layout.meta["generators"] == [1, 3]


def test_title_reports_girth_and_gap():
    fig = plot_evolution_spectrum(8, {1})  # Cay(Z_8, {1}) is the 8-cycle
    assert "girth 8" in fig.layout.title.text


def test_legend_hidden_for_large_vertex_count():
    fig = plot_evolution_spectrum(16, {1, 3})
    assert fig.layout.showlegend is False


def test_legend_shown_for_small_vertex_count():
    fig = plot_evolution_spectrum(6, {1})
    assert fig.layout.showlegend is True


def test_freq_to_color_endpoints():
    assert _freq_to_color(0.0).startswith("rgb(")
    assert _freq_to_color(1.0).startswith("rgb(")


def test_invalid_group_order_raises_stateviz_error():
    with pytest.raises(StateVizError):
        plot_evolution_spectrum(2, {1})


def test_require_eigenlight_noop_when_installed():
    # qiskit-eigenlight is installed in this test environment, so this
    # should not raise.
    _require_eigenlight()
