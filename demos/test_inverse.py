import numpy as np
from sensray import PlanetModel
from fwdop import GFwdOp, make_scalar_field

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

# Generate source and receiver
source_lat, source_lon, source_depth = 0.0, 0.0, 150.0  # Equator, 150 km depth
receiver_lat, receiver_lon = 30.0, 45.0  # Surface station

# Load G sparse matrix from file
G = GFwdOp(filepath="M1_sp_G_mat.npz")

# Generate different models and calculate dv
functions = {
    "simple": {"R": lambda r: np.ones_like(r), "T": lambda theta, phi: np.ones_like(theta)},
    "complex": {"R": lambda r: r**2 * np.exp(-r/100000), "T": lambda theta, phi: np.cos(theta)},
    "harmonic": {"R": lambda r: 0.1 * model.get_property_at_radius(radius=r, property_name="vp"), "T": lambda theta, phi: 0.5 * np.sqrt(5 / np.pi) * (3 * np.cos(theta) ** 2 - 1)},
}

func = "harmonic"
f = make_scalar_field(functions[func]["R"], functions[func]["T"])

model.mesh.project_function_on_mesh(f, property_name="dv")
print("Cell data 'dv':", model.mesh.mesh.cell_data["dv"])
model.mesh.plot_cross_section(source_lat=source_lat, source_lon=source_lon, receiver_lat=receiver_lat, receiver_lon=receiver_lon, property_name="dv")

travel_times = G(model.mesh.mesh.cell_data["dv"])
print(travel_times)

# adjoint correct if G.adjoint(v) = first kernel (G._K)
kernel_choice = 0  # desired kernel
print(G._K.shape[0])
v = np.zeros(G._K.shape[0])
v[kernel_choice] = 1
print(G.adjoint(v))
print(list(G._K.values())[kernel_choice])
