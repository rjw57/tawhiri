"""
Common utility functions for APIs.

"""
import numpy as np
import pyproj

WGS84_PROJECTION = pyproj.Proj(init='epsg:4326')
GEOCENT_PROJECTION = pyproj.Proj(proj='geocent', datum='WGS84', units='m')

def solver_to_geocentric(observations):
    """Take a Nx4 array of time, latitude, longitude, altitude observations and
    convert it to a Nx4 array of time, x, y and z geo-centric co-ordinates.
    Time is measured in seconds since the epoch and x, y and z are measured in
    metres.

    """
    output = np.vstack((observations[:, 0],) + pyproj.transform(
        WGS84_PROJECTION, GEOCENT_PROJECTION,
        observations[:, 2], observations[:, 1], observations[:, 3]
    )).T
    return output

def geocentric_to_wgs84(observations):
    """Take a Nx4 array of time, x, y, z observations and convert it to a Nx4
    array of time, longitude, latitude and altitude geo-centric co-ordinates.
    Time is measured in seconds since the epoch and x, y and z are measured in
    metres. NOTE ordering of longitude and latitude. This function is *NOT* the
    inverse of solver_to_geocentric() due to the re-ordering of longitude and
    latitude.

    """
    output = np.vstack((observations[:, 0],) + pyproj.transform(
        GEOCENT_PROJECTION, WGS84_PROJECTION,
        observations[:, 1], observations[:, 2], observations[:, 3]
    )).T
    return output
