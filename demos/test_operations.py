import numpy as np
from itertools import product
from sensray import PlanetModel
from fwdop import GFwdOp, make_scalar_field


# function to make points
def point(pointType="Source", minLat=-90, maxLat=90, minLon=-180, maxLon=180, minDepth=0, maxDepth=700):
    '''
    pointType: "Source" or "Receiver"
    For Source: lat, lon, depth
    For Receiver: lat, lon
    returns: (lat, lon, depth) or (lat, lon)'''
    if pointType == "Source":
        lat = np.random.uniform(minLat, maxLat)
        lon = np.random.uniform(minLon, maxLon)
        depth = np.random.uniform(minDepth, maxDepth)  # depth in km
        return (lat, lon, depth)
    elif pointType == "Receiver":
        lat = np.random.uniform(minLat, maxLat)
        lon = np.random.uniform(minLon, maxLon)
        return (lat, lon)
    else:
        raise ValueError("pointType must be 'Source' or 'Receiver'")


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


def make_list(a, b, n, x):
    """
    Create a list of length n with all elements equal to `a`,
    except the last x elements which are equal to `b`.
    """
    if x > n:
        raise ValueError("x cannot be greater than n")
    return [a] * (n - x) + [b] * x


# Load model and create mesh
model_name = "M1"
mesh_size_km = 1000


# model = PlanetModel.from_stand()
# # Create mesh and save if not exist, otherwise load existing
mesh_path = "M1_mesh"

# Load model and create mesh
model = PlanetModel.from_standard_model('M1')


try:
    model.create_mesh(from_file=mesh_path)
    print(f"Loaded existing mesh from {mesh_path}")
except FileNotFoundError:
    print("Creating new mesh...")
    radii = model.get_discontinuities()
    H_layers = make_list(a=mesh_size_km, b=mesh_size_km/5, n=len(radii), x=1)
    model.create_mesh(mesh_size_km=mesh_size_km, radii=radii, H_layers=H_layers)
    model.mesh.populate_properties(model.get_info()["properties"])
    model.mesh.save(f"{model_name}_mesh")  # Save mesh to VT
print(f"Created mesh: {model.mesh.mesh.n_cells} cells")


# make function for velocity perturbation
# Define R(r) and T(theta, phi)
R = lambda r: r**2 * np.exp(-r/100000)              # simple radial function
T = lambda theta, phi: np.cos(theta)         # angular dependence

R = lambda r: np.ones_like(r)
T = lambda theta, phi: np.ones_like(theta)

f = make_scalar_field(R, T)


# Generate source and receiver points and combinations
# sources = [point("Source", minDepth=150, maxDepth=150) for _ in range(2)]
# receivers = [point("Receiver", maxDepth=0) for _ in range(5)]
# phases = ["P", "S", "ScS"]
# srp = [prod + tuple([phases]) for prod in product(sources, receivers)]

# testing with one source-receiver pair - same as initial test
source_lat, source_lon, source_depth = 0.0, 0.0, 150.0  # Equator, 150 km depth
receiver_lat, receiver_lon = 30.0, 45.0  # Surface station
srp = [((source_lat, source_lon, source_depth), (receiver_lat, receiver_lon), ["P"])]

srr = get_rays(srp)

# print("Calculate travel time kernels and residuals...")
G = GFwdOp(model, srr[:,2])
# print(travel_times)

# integrate a function over a cell from the mesh
print("Integrating over cell...")
model.mesh.project_function_on_mesh(f, property_name="dv")

# model.mesh.mesh.cell_data["dv"] = integrals
print("Cell data 'dv':", model.mesh.mesh.cell_data["dv"])

travel_times = G(model.mesh.mesh.cell_data["dv"])

# display dv using first source-receiver pair
# model.mesh.display_dv(srr[0,0][0], srr[0,0][1], srr[0,1][0], srr[0,1][1], property_name="dv")


m = G.adjoint(travel_times)
