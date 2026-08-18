"""qiskit-stateviz: interactive Plotly-based statevector visualizations for Qiskit.

Native replacements for the static matplotlib views in
``qiskit.visualization`` (``plot_state_qsphere``, ``plot_bloch_multivector``),
built directly against current Qiskit ``Statevector`` / ``DensityMatrix``
objects and Qiskit SDK v2.x — no dependency on deprecated packages like
``qiskit-terra`` or ``qiskit-ibmq-provider``.
"""

from .bloch import plot_bloch_multivector_interactive
from .qsphere import plot_qsphere_interactive
from .rotation import plot_spin_rotation_interactive
from .utils import StateVizError

__version__ = "0.1.0"

__all__ = [
    "plot_qsphere_interactive",
    "plot_bloch_multivector_interactive",
    "plot_spin_rotation_interactive",
    "StateVizError",
]

# plot_evolution_spectrum depends on the optional qiskit-eigenlight package
# (`pip install qiskit-stateviz[spectrum]`). Importing qiskit_stateviz
# itself must never fail just because that extra isn't installed, so the
# import is guarded here rather than done unconditionally like the ones
# above.
try:
    from .spectrum import plot_evolution_spectrum  # noqa: F401

    __all__.append("plot_evolution_spectrum")
except ImportError:
    pass