import numpy as np
from scipy.sparse import csr_matrix

from fwdop.g_fwd_op import GFwdOp


class DummyMesh:
    def __init__(self, rows):
        # rows: list of 1D numpy arrays representing kernel rows
        self._rows = rows

    def compute_sensitivity_kernel(self, ray, **kwargs):
        # ray is just an index into rows for our test
        return self._rows[ray]


class DummyRay:
    def __init__(self, phase_name):
        class P:
            def __init__(self, name):
                self.name = name

        self.phase = P(phase_name)


def test_gfwdop_apply_and_adjoint():
    # Build a small synthetic kernel (3 obs x 4 voxels)
    rows = [
        np.array([1.0, 0.0, 2.0, 0.0]),
        np.array([0.0, 3.0, 0.0, 4.0]),
        np.array([5.0, 0.0, 0.0, 6.0]),
    ]

    mesh = DummyMesh(rows)

    # Create GFwdOp instance using mesh and fake rays. Use ray indices by
    # storing them inside a simple object with expected attributes. We abuse
    # the index order here for the test.
    class Model:
        def __init__(self, mesh, rows):
            self.mesh = mesh

    model = Model(mesh, rows)

    # We must pass rays that allow kernel lookup: our DummyMesh expects an
    # integer ray index. Create a tiny wrapper that carries both phase and an
    # index attribute.
    class LookupRay:
        def __init__(self, idx, phase_name):
            self.idx = idx

            class P:
                def __init__(self, name):
                    self.name = name

            self.phase = P(phase_name)

    lookup_rays = [LookupRay(0, "P"), LookupRay(1, "S"), LookupRay(2, "P")]

    # Monkeypatch compute_sensitivity_kernel to accept our LookupRay by
    # using idx
    def compute_kernel(ray, **kwargs):
        return rows[ray.idx]

    mesh.compute_sensitivity_kernel = compute_kernel

    op = GFwdOp(model, lookup_rays)

    # test apply: K @ x
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = op(x) if callable(op) else op.__call__(x)

    # expected multiply
    K = csr_matrix(np.vstack(rows))
    expected_y = K.dot(x)
    assert np.allclose(y, expected_y)

    # test adjoint: K.T @ y
    y_vec = np.array([1.0, 2.0, 3.0])
    # call adjoint method if available else __class__ fallback
    if hasattr(op, "adjoint"):
        ay = op.adjoint(y_vec)
    else:
        ay = op.__class__.adjoint(op, y_vec)
    expected_ax = K.T.dot(y_vec)
    assert np.allclose(ay, expected_ax)
