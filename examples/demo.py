"""Quick demo: interactive Q-sphere and Bloch multivector for a GHZ state.

Run with:
    python examples/demo.py

Requires pylatexenc in addition to the package's own dependencies, for
qc.draw('mpl'):
    pip install pylatexenc

Each figure opens in your default browser (or renders inline if run from
a Jupyter notebook / Colab cell instead of as a script).
"""

"""Quick demo: interactive Q-sphere and Bloch multivector.

Run with:
    python examples/demo.py

Requires pylatexenc in addition to the package's own dependencies, for
qc.draw('mpl'):
    pip install pylatexenc

Each figure opens in your default browser (or renders inline if run from
a Jupyter notebook / Colab cell instead of as a script).
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_stateviz import plot_bloch_multivector_interactive, plot_qsphere_interactive


def demo_ghz():
    """GHZ state: only 2 of 8 basis states populated, both at the poles.

    Good first example — simple to reason about, and it makes the
    "Bloch view can't see entanglement" point clearly since every
    single-qubit Bloch vector collapses to zero length.
    """
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)  # (|000> + |111>) / sqrt(2)
    sv = Statevector(qc)

    print("Rendering GHZ Q-sphere...")
    plot_qsphere_interactive(sv, title="GHZ state — Q-sphere").show()

    print("Rendering GHZ Bloch multivector...")
    plot_bloch_multivector_interactive(sv, title="GHZ state — Bloch multivector").show()

    print(
        "Notice: each qubit's individual Bloch vector is (near) zero-length "
        "despite the state being pure — the Bloch view can't see the "
        "entanglement that the Q-sphere shows clearly via the two populated "
        "poles (|000> and |111>)."
    )


def demo_rotation_showcase():
    """States chosen specifically to need rotation, unlike GHZ (which only
    populates the two poles — rotating around z shows nothing new there).
    """
    # Q-sphere: all 8 basis states populated, spread across every
    # Hamming-weight ring (0 through 3) with varied phases. Points sit at
    # different azimuthal angles on the weight-1 and weight-2 rings, so a
    # single fixed camera angle can't show all of them — you have to drag
    # to orbit around to see the far side of each ring.
    qc_q = QuantumCircuit(3)
    qc_q.h(0)
    qc_q.h(1)
    qc_q.h(2)
    qc_q.t(0)
    qc_q.s(1)
    qc_q.cz(1, 2)
    sv_q = Statevector(qc_q)

    print("Rendering rotation-showcase Q-sphere (all 8 basis states, all 4 rings)...")
    plot_qsphere_interactive(sv_q, title="All basis states populated — rotate to see every ring").show()

    # Bloch multivector: three independent single-qubit rotations, each
    # landing at a genuinely 3D (non-axis-aligned) point. None of the
    # three vectors are coplanar, so no single camera angle shows all
    # three directions clearly — a good case for the free 360° orbit.
    qc_b = QuantumCircuit(3)
    qc_b.u(1.0, 0.3, 0.7, 0)
    qc_b.u(2.0, 1.1, 0.2, 1)
    qc_b.u(0.5, 2.4, 1.8, 2)
    sv_b = Statevector(qc_b)

    print("Rendering rotation-showcase Bloch multivector (3 off-axis vectors)...")
    plot_bloch_multivector_interactive(
        sv_b, title="Three independent, off-axis Bloch vectors — rotate each sphere"
    ).show()


def main():
    demo_ghz()
    demo_rotation_showcase()


if __name__ == "__main__":
    main()
