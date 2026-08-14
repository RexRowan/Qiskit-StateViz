from .bloch import plot_bloch_multivector_interactive
 from .qsphere import plot_qsphere_interactive
+from .rotation import plot_spin_rotation_interactive
 from .utils import StateVizError

 __version__ = "0.1.0"

 __all__ = [
     "plot_qsphere_interactive",
     "plot_bloch_multivector_interactive",
+    "plot_spin_rotation_interactive",
     "StateVizError",
 ]