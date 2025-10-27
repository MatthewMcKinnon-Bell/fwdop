import numpy as np
import matplotlib.pyplot as plt

def plot_shell_property(radii, f_j, *,
                        show_shading=True,
                        show_centers=True,
                        annotate_radii=True,
                        figsize=(8,4),
                        title="Shell property on 1D spherical mesh"):
    """
    Plot a property defined per-shell on a 1D spherical radial discretization.

    Parameters
    ----------
    radii : 1D array-like, length N+1
        Shell edges: r_0, r_1, ..., r_N
    f_j : 1D array-like, length N
        Property value for each shell [r_j, r_{j+1})
    show_shading : bool
        Shade shells to make discretization clearer
    show_centers : bool
        Show markers at shell centers
    annotate_radii : bool
        Annotate radii values on the plot (may clutter if many shells)
    """
    radii = np.asarray(radii, dtype=float)
    f_j = np.asarray(f_j, dtype=float)
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError("radii must be 1D array with length >= 2")
    if f_j.ndim != 1 or f_j.size != radii.size - 1:
        raise ValueError("f_j must have length len(radii)-1")

    # Prepare step plot data: piecewise-constant
    # Use stairs/step plotting: create repeated arrays for step plotting
    # Approach: r_steps of length 2N, v_steps length 2N with v_j repeated
    N = f_j.size
    r_steps = np.empty(2 * N)
    v_steps = np.empty(2 * N)
    r_steps[0::2] = radii[:-1]
    r_steps[1::2] = radii[1:]
    v_steps[0::2] = f_j
    v_steps[1::2] = f_j

    centers = 0.5 * (radii[:-1] + radii[1:])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Step plot: use drawstyle='steps-post' or stairs if available
    # Using stairs (clear intent) if matplotlib >= 3.4; otherwise fallback to step
    try:
        ax.stairs(f_j, radii, baseline=None, label="shell property (piecewise-constant)")
    except Exception:
        ax.plot(r_steps, v_steps, drawstyle="steps-post", label="shell property (piecewise-constant)")

    # Vertical lines for shell edges
    for r in radii:
        ax.axvline(r, color="0.85", linewidth=0.8, zorder=0)

    # Optional shading of shells for visibility
    if show_shading:
        cmap = plt.get_cmap("Blues")
        # Normalize to property range for shading intensity (avoid division by zero)
        vmin, vmax = np.min(f_j), np.max(f_j)
        rng = vmax - vmin if vmax != vmin else 1.0
        for i in range(N):
            r0, r1 = radii[i], radii[i+1]
            # shade using normalized value -> subtle color
            norm_val = (f_j[i] - vmin) / rng
            ax.axvspan(r0, r1, facecolor=cmap(0.2 + 0.6 * norm_val), alpha=0.08, zorder=0)

    # Centers markers
    if show_centers:
        ax.plot(centers, f_j, "o", markersize=4, label="shell centers")

    # Annotate radii if requested
    if annotate_radii:
        for r in radii:
            ax.text(r, ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                    f"{r:.3g}", rotation=90, va="top", ha="center", fontsize=8)

    ax.set_xlabel("radius (r)")
    ax.set_ylabel("property value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig, ax


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Example: non-uniform radii (length N+1)
    radii = np.concatenate(([0.0], np.linspace(0.01, 1.0, 12)))
    # Example shell property: e.g., from formula or computed f_j
    # Here we create a synthetic property (e.g., average of f(r)=exp(-r) within shell)
    def f(r):
        return np.exp(-r) * (1 + 0.5 * r)

    # compute simple trapz per-shell (for example only)
    f_j = np.empty(len(radii)-1)
    for i in range(len(radii)-1):
        r0, r1 = radii[i], radii[i+1]
        rs = np.linspace(r0, r1, 9)
        integrand = f(rs) * rs**2
        integral = np.trapz(integrand, rs)
        denom = r1**3 - r0**3
        f_j[i] = 3.0 * integral / denom if denom != 0 else f(0.5*(r0+r1))

    fig, ax = plot_shell_property(radii, f_j, show_shading=True, show_centers=True)
    plt.show()


'''


import numpy as np
from sensray import PlanetModel
from fwdop import GFwdOp, make_scalar_field
from pygeoinf.linear_solvers import LUSolver, CholeskySolver
from ray_and_point_generation import get_rays, fibonacci_sphere_points
from itertools import product
from random import randint
import math
from itertools import chain
from obspy.geodetics import locations2degrees







# ------------------------------------------------------------------------------------------------


model = PlanetModel.from_standard_model('M1')

r_max = model.radius
r_min = 0
n_shells = 10
shell_radii = np.linspace(r_max, r_min, n_shells + 1)
print(shell_radii)

# ------------------------------------------------------------------------------------------------

import numpy as np
import warnings
from typing import Callable, Tuple

try:
    from scipy import integrate
except Exception as e:
    raise ImportError("scipy is required: pip install scipy") from e

FOUR_PI = 4.0 * np.pi
THREE = 3.0

def _gauss_legendre_shell_integrals(
    f_vec: Callable[[np.ndarray], np.ndarray],
    radii: np.ndarray,
    npts: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Gauss-Legendre evaluation of I_j = ∫_{r_j}^{r_{j+1}} f(r) r^2 dr
    Returns (I_j array, denom array where denom = r_{j+1}^3 - r_j^3)
    """
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError("radii must be a 1D array of length >= 2")

    xi, wi = np.polynomial.legendre.leggauss(npts)  # nodes & weights on [-1,1]
    rL = radii[:-1]
    rR = radii[1:]
    J = 0.5 * (rR - rL)       # mapping scale
    r_mid = 0.5 * (rR + rL)

    # r_eval shape (N_shells, npts)
    r_eval = r_mid[:, None] + J[:, None] * xi[None, :]

    # evaluate f at all points (expects vectorized f)
    fvals = f_vec(r_eval)
    fvals = np.asarray(fvals)
    if fvals.shape != r_eval.shape:
        # try broadcast shapes (npts,) -> (N, npts) etc.
        try:
            fvals = np.broadcast_to(fvals, r_eval.shape)
        except Exception:
            raise ValueError("f_vec did not return an array of shape matching r_eval")

    # integral per shell
    weighted = (wi[None, :] * fvals * (r_eval ** 2)) * J[:, None]
    integrals = np.sum(weighted, axis=1)  # ∫ f(r) r^2 dr over each shell (no 4π factor)
    denom = (rR ** 3) - (rL ** 3)
    return integrals, denom


def _quad_shell_integrals(
    f_scalar: Callable[[float], float],
    radii: np.ndarray,
    epsabs: float = 1e-9,
    epsrel: float = 1e-9,
    limit: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive quad per shell. f_scalar should accept float and return float.
    Returns (integrals, denom) where integrals are ∫ f(r) r^2 dr.
    """
    radii = np.asarray(radii, dtype=float)
    rL = radii[:-1]
    rR = radii[1:]
    N = rL.size
    integrals = np.empty(N, dtype=float)

    for i in range(N):
        a = rL[i]
        b = rR[i]
        if b <= a:
            integrals[i] = 0.0
            continue
        val, err = integrate.quad(lambda r: float(f_scalar(r)) * (r ** 2), a, b,
                                  epsabs=epsabs, epsrel=epsrel, limit=limit)
        integrals[i] = val

    denom = (rR ** 3) - (rL ** 3)
    return integrals, denom


# === Method to include in your class ===
def project_to_spherical_shells(
    self,
    function: Callable,
    property_name: str,
    method: str = "gauss",  # 'gauss' | 'quad'
    npts: int = 12,
    quad_opts: dict = None,
):
    """
    Project a radial function onto 1D spherical shells using formula:
      f_j = 3 * ∫_{r_j}^{r_{j+1}} f(r) r^2 dr / (r_{j+1}^3 - r_j^3)

    - function: callable, ideally vectorized f(r) -> shape-matching array
    - property_name: name to store in shell_grid.cell_data[property_name]
    - radii: optional array of radii length N+1; if None, tries self.shell_radii
    - method: 'gauss' for vectorized Gauss-Legendre, 'quad' for scipy.quad
    - npts: Gauss points for 'gauss' (or fixed_quad fallback)
    - quad_opts: dict passed to _quad_shell_integrals (epsabs, epsrel, limit)
    """
    # 1) Compute integrals per shell ∫ f(r) r^2 dr and denom = r_{j+1}^3 - r_j^3
    if method == "gauss":
        # prefer vectorized evaluation
        try:
            integrals, denom = _gauss_legendre_shell_integrals(function, self.mesh.radii, npts=npts)
        except Exception as e:
            # if vectorized evaluation fails, fall back to scalar quad with warning
            warnings.warn(f"Gauss vectorized evaluation failed ({e}); falling back to adaptive quad.", UserWarning)
            quad_opts = quad_opts or {}
            integrals, denom = _quad_shell_integrals(function, self.mesh.radii, **quad_opts)
    elif method == "quad":
        quad_opts = quad_opts or {}
        integrals, denom = _quad_shell_integrals(function, self.mesh.radii, **quad_opts)
    else:
        raise ValueError("method must be 'gauss' or 'quad'")

    # 2) compute shell-averaged f_j
    # handle degenerate denom (zero-volume shells)
    small = denom == 0.0
    f_j = np.empty_like(denom, dtype=float)
    # formula: f_j = 3 * integral / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        f_j[~small] = THREE * integrals[~small] / denom[~small]

    if np.any(small):
        # For zero-thickness shells fallback to midpoint value of function
        mids = 0.5 * (self.mesh.radii[:-1][small] + self.mesh.radii[1:][small])
        # attempt vectorized call; if it fails fall back to scalar
        try:
            f_j[small] = np.asarray(function(mids))
        except Exception:
            f_j[small] = np.array([function(float(m)) for m in mids])

    """
    # 4) Optionally compute volumes (useful if you want cell volumes like tetra code)
    volumes = FOUR_PI * (denom / 3.0)  # since V_shell = 4π/3 (rR^3 - rL^3)

    # 5) Store into grid-like object similar to your tetra code:
    #    try to find self.shell_grid (preferred), else self.grid_shells, else attach attribute
    target_grid = None
    for attr in ("shell_grid", "shells", "shell_grid_obj", "grid_shells"):
        if hasattr(self, attr):
            target_grid = getattr(self, attr)
            break

    if target_grid is None:
        # fallback: attach to 'self' as a simple container with cell_data dict
        # Create a minimal object to hold cell_data if needed
        class _Tmp:
            pass
        target_grid = getattr(self, "shell_grid", None)
        if target_grid is None:
            tmp = _Tmp()
            tmp.cell_data = {}
            target_grid = tmp
            # attach back so subsequent calls reuse it
            self.shell_grid = target_grid
    """
    # store arrays as float32 to match your tetra code
    # f_j has length N (number of shells)
    model.mesh.cell_data[property_name] = np.asarray(f_j, dtype=np.float32)
    """
    # optionally also store volumes and denom if desired
    target_grid.cell_data[property_name + "_volume"] = np.asarray(volumes, dtype=np.float64)
    target_grid.cell_data[property_name + "_denom"] = np.asarray(denom, dtype=np.float64)

    # return computed arrays for immediate use
    return f_j, denom, volumes
    """



# ----------------------------------------------------
# Shell utilities
# ----------------------------------------------------
def build_shell_radii(r_min, r_max, n_shells):
    return np.linspace(r_max, r_min, n_shells + 1)

def find_shell_index(r, shell_radii):
    for i in range(len(shell_radii) - 1):
        if shell_radii[i] >= r > shell_radii[i + 1]:
            return i
    return None

def interpolate_t(r0, r1, target):
    if r1 == r0:
        return 0.0
    return (target - r0) / (r1 - r0)


# ----------------------------------------------------
# Shell marching kernel
# ----------------------------------------------------
def compute_shell_contributions(r0, th0, r1, th1, contrib):
    shell0 = find_shell_index(r0, shell_radii)
    shell1 = find_shell_index(r1, shell_radii)

    if shell0 is None or shell1 is None:
        return

    L_full = np.sqrt((r1 - r0)**2 + (model.radius * (th1 - th0))**2)

    if shell0 == shell1:
        contrib[shell0] += L_full
        return

    direction = int(np.sign(r0 - r1))
    boundaries = []

    if direction > 0:  # inward
        for rad in shell_radii[shell0 + 1:shell1 + 1]:
            boundaries.append(rad)
    else:  # outward
        for rad in shell_radii[shell1:shell0]:
            boundaries.append(rad)

    boundaries.append(r1)

    prev_r, prev_th = r0, th0
    current_shell = shell0

    for boundary_r in boundaries:
        t = np.clip(interpolate_t(r0, r1, boundary_r), 0, 1)
        interp_r = r0 + (r1 - r0) * t
        interp_th = th0 + (th1 - th0) * t

        L = np.sqrt((interp_r - prev_r)**2 +
                    (model.radius * (interp_th - prev_th))**2)

        contrib[current_shell] += L

        prev_r, prev_th = interp_r, interp_th
        current_shell += direction

def _ensure_shell_edges(shell_radii):
    shell_radii = np.asarray(shell_radii, dtype=float)
    if shell_radii.ndim != 1 or shell_radii.size < 2:
        raise ValueError("shell_radii must be 1D with at least two entries")
    was_desc = False
    if shell_radii[0] > shell_radii[-1]:
        shell_edges = shell_radii[::-1].copy()   # ascending internally
        was_desc = True
    else:
        shell_edges = shell_radii.copy()
    return shell_edges, was_desc

def avg_velocity_per_shell_from_radius(model, 
                                       property_name='vp', 
                                       n_quad=24):
    """
    Compute average velocity in each shell using a radius-based accessor.
    Assume shell radii are ascending

    Parameters
    ----------
    property_name : str
        Property to request, e.g. 'vp' or 'vs'.
    n_quad : int
        Number of Gauss-Legendre quadrature points per shell.

    Returns
    -------
    avg_v : ndarray, shape (n_shells,)
        Average velocity in each shell in the same order as the provided shell_radii.
    """
    shell_edges, was_desc = _ensure_shell_edges(model.mesh.shell_radii)
    n_shells = len(shell_edges) - 1
    avg_vs = np.zeros(n_shells, dtype=float)

    # Gauss-Legendre nodes/weights on [-1,1]
    xg, wg = np.polynomial.legendre.leggauss(n_quad)

    for j in range(n_shells):
        r0 = shell_edges[j]
        r1 = shell_edges[j + 1]
        if r1 <= r0:
            raise ValueError("shell edges must be strictly increasing internally")
        # map nodes -> [r0, r1]
        nodes = 0.5 * (r1 - r0) * xg + 0.5 * (r1 + r0)
        weights = 0.5 * (r1 - r0) * wg
        # evaluate property at nodes (vectorized)
        vals = model.get_prop_at_radius(property_name, nodes)   # expects radius in km
        # integrand is v(r) (we want (1/(r1-r0)) * ∫_{r0}^{r1} v(r) dr)
        integral = np.sum(vals * weights)
        avg_vs[j] = integral / (r1 - r0)

    # return in original ordering
    if was_desc:
        return avg_vs[::-1]
    else:
        return avg_vs


# ----------------------------------------------------
# Ray kernel wrapper
# ----------------------------------------------------
from typing import Optional, Dict, List, Any, TYPE_CHECKING, Callable


def compute_sensitivity_kernel(
        arrival: Any,
        property_name: str,
        attach_name: Optional[str] = None,
        epsilon: float = 1e-6,
        replace_existing: bool = True
    ) -> np.ndarray:
    # kernel is path lengths in each shell
    kernels = []

    path = arrival.path
    depth = path["depth"]
    dist = np.radians(path["dist"])
    radius = model.radius - depth

    lengths = np.zeros(len(shell_radii) - 1)
    
    #prop = project_to_spherical_shells(self=model, function=lambda , property_name=property_name)
    prop = model.get_property_at_radius(property_name, shell_radii)

    for i in range(len(radius) - 1):
        compute_shell_contributions(radius[i], dist[i],
                                    radius[i + 1], dist[i + 1], lengths)

    # add code to compute travel times for each length in the shells
    # K = -L/(v^2+epsilon)
    contrib = -lengths/(prop**2+epsilon)

    kernels.append((arrival.name, contrib))

    return kernels


# ----------------------------------------------------
# Build full G matrix
# ----------------------------------------------------
def build_G_matrix(rays):
    """
    events: list of (lat, lon, depth)
    stations: list of (lat, lon, depth)
    phases: list of phases to compute (e.g. ["P"])
    """
    rows = []
    labels = []

    for num, (ray) in enumerate(rays):
        results = compute_sensitivity_kernel(ray, property_name="vp")

        for phase, kernel in results:
            rows.append(kernel)
            labels.append((num, phase))

    G = np.vstack(rows)
    return G, labels


# ----------------------------------------------------
# Example
# ----------------------------------------------------
if __name__ == "__main__":
    # Generate sources and receivers
    setup_info = {
        "source": {"N": 5, "min depth": 150, "max depth": 150},
        "receiver": {"N": 5, "min depth": 0, "max depth": 0},
    }
    depth = randint(setup_info["source"]["min depth"], setup_info["source"]["max depth"])
    sources = fibonacci_sphere_points(setup_info["source"]["N"], radius=model.radius-depth, latlon=True)  # 20 sources at 150km depth
    receivers = fibonacci_sphere_points(setup_info["source"]["N"], radius=model.radius, latlon=True)  # 20 stations on Earth radius
    phases = ["P"]

    srr = get_rays(model=model, srp=product(sources, receivers, phases), radius=True)
    print(srr)

    # G = GFwdOp(model, srr[:, 2])
    G, meta = build_G_matrix(srr[:,2])

    print("G matrix:\n", G)
    print("Row metadata:", meta)
'''