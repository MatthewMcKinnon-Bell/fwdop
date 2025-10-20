import numpy as np
from typing import Callable, Union


Number = Union[float, int]


def make_scalar_field(
    R: Callable[[np.ndarray], np.ndarray],
    T: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Callable[[Union[np.ndarray, list, tuple, Number], Union[Number, None], Union[Number, None]], np.ndarray]:
    """
    Creates a scalar field f(x, y, z) = R(r) * T(theta, phi),
    vectorized to work with quadpy (input shape (3, n_points)).

    Parameters
    ----------
    R : callable
        Radial function R(r)
    T : callable
        Angular function T(theta, phi)

    Returns
    -------
    f : callable
        Vectorized scalar field f(x, y, z)
    """

    def f(x, y=None, z=None):
        # Case 1: quadpy-style call, x is (3, n_points)
        if y is None and z is None:
            x = np.asarray(x)
            # Accept (3, n_points) or (n_points, 3)
            if x.ndim == 2 and x.shape[0] == 3:
                X, Y, Z = x[0], x[1], x[2]
            elif x.ndim == 2 and x.shape[1] == 3:
                X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
            elif x.shape == (3,):
                X, Y, Z = x
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        else:
            # Case 2: standard call f(x, y, z)
            X = np.asarray(x)
            Y = np.asarray(y)
            Z = np.asarray(z)

        # Convert Cartesian to spherical
        r = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.where(r == 0, 0.0, np.arccos(np.clip(Z / r, -1.0, 1.0)))
        phi = np.mod(np.arctan2(Y, X), 2*np.pi)

        return R(r) * T(theta, phi)

    return f