# VisTools

**VisTools** is a utility library for scientific visualization workflows, focused on the pure VTK data format as well as PyVista.

## Modules

- `vistools.vtk`: Utility functions based only on the plain `vtk` python package.

- `vistools.pyvista`: Utility functions for PvVista, depends on `vistools.vtk`.

## Installation

VisTools is available via `pip` and can be installed with
```bash
# Install only vistools.vtk
pip install vistools
# Install all modules
pip install "vistools[pyvista]"
```
