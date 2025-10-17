# fwdop

Small package containing the GFwdOp forward-operator utilities used in SensRay.

This package provides a simple `GFwdOp` class that wraps a stacked sparse
kernel matrix and exposes a mapping/adjoint interface.

Install locally for development:

```bash
cd fwdop
pip install -e .
```

Run tests:

```bash
pytest -q
```

Examples
--------

1) Quick import

```python
from fwdop import GFwdOp
print(GFwdOp)
```

2) Synthetic example (equivalent to the unit test)

```python
import numpy as np
from scipy.sparse import csr_matrix
from fwdop import GFwdOp

# Create a tiny in-memory mesh-like object for testing
class DummyMesh:
	def __init__(self, rows):
		self._rows = rows
	def compute_sensitivity_kernel(self, ray, **kwargs):
		return self._rows[ray.idx]

class LookupRay:
	def __init__(self, idx, phase_name):
		self.idx = idx
		class P:
			def __init__(self, name):
				self.name = name
		self.phase = P(phase_name)

rows = [
	np.array([1.0, 0.0, 2.0, 0.0]),
	np.array([0.0, 3.0, 0.0, 4.0]),
	np.array([5.0, 0.0, 0.0, 6.0]),
]

mesh = DummyMesh(rows)
model = type("M", (), {"mesh": mesh})()
rays = [LookupRay(0, "P"), LookupRay(1, "S"), LookupRay(2, "P")]

op = GFwdOp(model, rays)

x = np.array([1.0, 2.0, 3.0, 4.0])
print("Kx =", op(x))
```

3) Minimal SensRay integration sketch

```python
# Given a SensRay model object with a .mesh that can compute kernels:
from fwdop import GFwdOp

# `rays` should be a sequence of objects with `phase.name` and any other
# attributes your model's compute_sensitivity_kernel expects.
op = GFwdOp(sensray_model, rays)
# then use op like any linear operator (apply to voxel vector, use adjoint)
```

