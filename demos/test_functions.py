import numpy as np
from sensray import PlanetModel
from fwdop import GFwdOp, make_scalar_field


def get_rays(srp):
    '''
    srp: list of tuples (source, receiver, phases)
    where source = (lat, lon, depth), receiver = (lat, lon), phases = [phase1, phase2, ...]
    returns array of (source, receiver, ray) for each ray
    '''
    srr_list = []
    for (source, receiver, phases) in srp:
        rays = model.taupy_model.get_ray_paths_geo(
            source_depth_in_km=source[2],
            source_latitude_in_deg=source[0],
            source_longitude_in_deg=source[1],
            receiver_latitude_in_deg=receiver[0],
            receiver_longitude_in_deg=receiver[1],
            phase_list=phases,
        )
        for ray in rays:
            srr_list.append((source, receiver, ray))

    return np.array(srr_list, dtype=object)


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


# Generate srr and compute G from rays
source_lat, source_lon, source_depth = 0.0, 0.0, 150.0  # Equator, 150 km depth
receiver_lat, receiver_lon = 30.0, 45.0  # Surface station
srp = [((source_lat, source_lon, source_depth), (receiver_lat, receiver_lon), ["P"])]

srr = get_rays(srp)

# G = GFwdOp(model, srr[:,2])
# G.save_matrix("M1_sp_G_mat.npz")

G = GFwdOp(NPZ_filepath="M1_sp_G_mat.npz")

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
model.mesh.display_dv(source_lat=source_lat, source_lon=source_lon, receiver_lat=receiver_lat, receiver_lon=receiver_lon, property_name="dv")

travel_times = G(model.mesh.mesh.cell_data["dv"])
print(travel_times)