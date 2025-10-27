import numpy as np
import matplotlib.pyplot as plt

def plot_shell_property(radii, f_j, *,
                        show_shading=True,
                        show_centers=True,
                        annotate_radii=False,
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