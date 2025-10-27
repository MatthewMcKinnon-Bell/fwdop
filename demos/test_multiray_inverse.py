import numpy as np
from sensray import PlanetModel
from fwdop import GFwdOp, make_scalar_field
from pygeoinf.linear_solvers import LUSolver, CholeskySolver
from ray_and_point_generation import get_rays, fibonacci_sphere_points
from itertools import product
from random import randint

# Load model and create mesh
model_name = "M1"
mesh_size_km = 1000

# Create mesh and save if not exist, otherwise load existing
mesh_path = "M1_mesh"

# Load model and create mesh
model = PlanetModel.from_standard_model('M1')


try:
    model.create_mesh(from_file=mesh_path)
    print(f"Loaded existing mesh from {mesh_path}")
except FileNotFoundError:
    print("Creating new mesh...")
    radii = model.get_discontinuities()
    H_layers = [1000, 600]
    model.create_mesh(mesh_size_km=mesh_size_km, radii=radii, H_layers=H_layers)
    model.mesh.populate_properties(model.get_info()["properties"])
    model.mesh.save(f"{model_name}_mesh")  # Save mesh to VT
print(f"Created mesh: {model.mesh.mesh.n_cells} cells")

# Generate sources and receivers
setup_info = {
    "source": {"N": 10, "min depth": 150, "max depth": 150},
    "receiver": {"N": 10, "min depth": 0, "max depth": 0},
}

# Example usage:
# get evenly distributed points at a single depth
depth = randint(setup_info["source"]["min depth"], setup_info["source"]["max depth"])
sources = fibonacci_sphere_points(setup_info["source"]["N"], radius=model.radius-depth, latlon=True)  # 20 sources at 150km depth
receivers = fibonacci_sphere_points(setup_info["source"]["N"], radius=model.radius, latlon=True)  # 20 stations on Earth radius
phases = ["P"]
print(sources)
print(receivers)
srr = get_rays(model=model, srp=product(sources, receivers, phases), radius=True)

# Load G sparse matrix from file
G = GFwdOp(model=model, rays=srr[:,2])

# Generate different models and calculate dv
functions = {
    "radial": {"R": lambda r: r, "T": lambda theta, phi: np.ones_like(theta)},
    "simple": {"R": lambda r: np.ones_like(r), "T": lambda theta, phi: np.ones_like(theta)},
    "complex": {"R": lambda r: r**2 * np.exp(-r/100000), "T": lambda theta, phi: np.cos(theta)},
    "harmonic": {"R": lambda r: 0.1 * model.get_property_at_radius(radius=r, property_name="vp"), "T": lambda theta, phi: 0.5 * np.sqrt(5 / np.pi) * (3 * np.cos(theta) ** 2 - 1)},
}

func = "radial"
f = make_scalar_field(functions[func]["R"], functions[func]["T"])

model.mesh.project_function_on_mesh(f, property_name="dv")
print("Cell data 'dv':", model.mesh.mesh.cell_data["dv"])

travel_times = G(model.mesh.mesh.cell_data["dv"])
print(travel_times)

# Inverse in a single operation
M_tilde = (G.adjoint@LUSolver()(G@G.adjoint))(G(model.mesh.mesh.cell_data["dv"]))
model.mesh.mesh.cell_data["solution"] = M_tilde
print(M_tilde)

print("Solution visualization...")
plotter1 = model.mesh.plot_cross_section(plane_normal=(0, 1, 0), property_name="solution")

plotter1.camera.position = (8000, 6000, 10000)

plotter1.show()
