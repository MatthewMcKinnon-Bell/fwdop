import math
import numpy as np
from itertools import chain


# Generate N evenly spaced points at given radius
def fibonacci_points_on_sphere(N, radius=1.0):
    """
    Generate N quasi-uniformly spaced points on the surface of a sphere
    using Fibonacci spiral method.
    Returns array of shape (N, 3)
    """
    points = np.zeros((N, 3))
    phi = (1 + 5**0.5) / 2  # golden ratio
    for i in range(N):
        z = 1 - (2*i + 1)/N
        theta = math.acos(z)
        az = 2 * math.pi * ((i / phi) % 1.0)
        r_xy = math.sqrt(1 - z**2)
        x = radius * r_xy * math.cos(az)
        y = radius * r_xy * math.sin(az)
        z = radius * z
        points[i] = [x, y, z]
    return points


# Generate N evenly spaced points at given radius
def fibonacci_sphere_points(N, radius=1.0, latlon=False):
    """
    Generate N quasi-uniformly spaced points on a sphere of given radius.

    Parameters
    ----------
    N : int
        Number of points.
    radius : float
        Radius of sphere (same units as desired output).
    latlon : bool
        If True, return (latitude_deg, longitude_deg, radius).
        If False, return (x, y, z) in Cartesian coordinates.

    Returns
    -------
    np.ndarray
        (N, 3) array of points:
        - (x, y, z) if latlon=False
        - (lat_deg, lon_deg, radius) if latlon=True
    """
    phi = (1 + 5**0.5) / 2  # golden ratio
    points = np.zeros((N, 3))
    
    for i in range(N):
        z = 1 - (2*i + 1)/N  # evenly spaced z-values in [-1,1]
        theta = math.acos(z)  # polar angle
        lon_rad = 2 * math.pi * ((i / phi) % 1.0)  # azimuth

        if latlon:
            # Convert to latitude and longitude in degrees
            lat_deg = math.degrees(math.asin(z))      # latitude from z
            lon_deg = math.degrees(lon_rad)
            if lon_deg > 180:
                lon_deg -= 360
            points[i] = [lat_deg, lon_deg, radius]
        else:
            # Convert to Cartesian coordinates
            r_xy = math.sqrt(1 - z*z)
            x = radius * r_xy * math.cos(lon_rad)
            y = radius * r_xy * math.sin(lon_rad)
            z = radius * z
            points[i] = [x, y, z]
    
    return points


# generate random point within ranges specified
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


def get_rays(model, srp, radius=False):
    """
    Accounts for the sources and receivers being passed in with radius instead of depth
    srp: list of tuples (source, receiver, phases)
    where source = (lat, lon, depth/radius), receiver = (lat, lon, depth/radius), phases = [phase1, phase2, ...]
    returns array of (source, receiver, ray) for each ray
    """
    return np.array(
        list(chain.from_iterable(map(
            lambda s_r_p: [
                (s_r_p[0], s_r_p[1], ray)
                for ray in model.taupy_model.get_ray_paths_geo(
                    source_depth_in_km=s_r_p[0][2] if not radius else model.radius-s_r_p[0][2],
                    source_latitude_in_deg=s_r_p[0][0],
                    source_longitude_in_deg=s_r_p[0][1],
                    receiver_latitude_in_deg=s_r_p[1][0],
                    receiver_longitude_in_deg=s_r_p[1][1],
                    phase_list=s_r_p[2],
                )
            ],
            srp
        ))),
        dtype=object
    )
