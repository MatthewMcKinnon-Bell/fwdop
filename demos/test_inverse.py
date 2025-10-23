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

# Inversion

# Test adjoint - Correct if G.adjoint(v) = first kernel (G._K)
kernel_choice = 0  # desired kernel
v = np.zeros(G._K.shape[0])
v[kernel_choice] = 1
print(f"Test Adjoint Success: {np.allclose(G.adjoint(v), G._K.getrow(kernel_choice).toarray().ravel())}")

# Compute lambda from G and G.adjoint (GG^T)
lambda_val = (G@G.adjoint)
# Test lambda - Correct if dense matrix is symmetric
print(f"Test lambda Success: {np.allclose(lambda_val.matrix(dense=True), lambda_val.matrix(dense=True).T)}")

# Test solver - Shouldn't raise an error of any kind
from pygeoinf.linear_solvers import LUSolver, CholeskySolver
solver = LUSolver()
lambda_inv = solver(lambda_val)

# G_dagger - Should compute cell inverse
G_dagger = G.adjoint@lambda_inv

# M_tilde - G_dagger(d) where d is G applied onto the model. Dims N voxels
M_tilde = G_dagger(travel_times)

model.mesh.mesh.cell_data["solution"] = M_tilde
# Test for one kernel - If M_tilde = kernel * G(m) / lambda
print((G.matrix(dense=True) * travel_times / lambda_val.matrix(dense=True))[0])
model.mesh.mesh.cell_data["kernel x d/lambda"] = (G.matrix(dense=True) * travel_times / lambda_val.matrix(dense=True))[0]

# Inverse in a single operation
M_tilde2 = (G.adjoint@solver(G@G.adjoint))(travel_times)
print(np.allclose(M_tilde, (G.matrix(dense=True) * travel_times / lambda_val.matrix(dense=True))[0]))

print("Solution visualization...")
plotter1 = model.mesh.plot_cross_section(source_lat=source_lat, source_lon=source_lon, receiver_lat=receiver_lat, receiver_lon=receiver_lon, property_name="kernel x d/lambda")

plotter1.camera.position = (8000, 6000, 10000)

plotter1.show()
