"""qiskit-stateviz: interactive Plotly-based statevector visualizations for Qiskit.

Native replacements for the static matplotlib views in
``qiskit.visualization`` (``plot_state_qsphere``, ``plot_bloch_multivector``),
built directly against current Qiskit ``Statevector`` / ``DensityMatrix``
objects and Qiskit SDK v2.x — no dependency on deprecated packages like
``qiskit-terra`` or ``qiskit-ibmq-provider``.
"""

from .bloch import plot_bloch_multivector_interactive
from .qsphere import plot_qsphere_interactive
from .utils import StateVizError

__version__ = "0.1.0"

__all__ = [
    "plot_qsphere_interactive",
    "plot_bloch_multivector_interactive",
    "StateVizError",
]
