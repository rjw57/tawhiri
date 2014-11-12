# Copyright 2014 (C) Rich Wareham <rich.cusf@richwareham.com>
#
# This file is part of Tawhiri.
#
# Tawhiri is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Tawhiri is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Tawhiri.  If not, see <http://www.gnu.org/licenses/>.
"""
Provide an experimental HTTP API for Tawhiri as a Flask Blueprint.

"""
from collections import namedtuple
from contextlib import contextmanager
import json
import random

from flask import Blueprint, jsonify, request, current_app
import numpy as np
from osgeo import ogr, gdal
import pyproj
from werkzeug.exceptions import BadRequest, NotFound

from tawhiri import solver, models
from tawhiri.dataset import Dataset as WindDataset
from ruaumoko import Dataset as ElevationDataset

api = Blueprint('api_experimental', __name__)

@api.route('/')
def index():
    return jsonify(dict(version=2))

@api.route('/predict', methods=['POST'])
def predict():
    # Parse request body
    prediction_spec = parse_prediction_spec_request()

    # Run predictions
    results = run_predictions(prediction_spec)

    # Compute some stats
    landing_mean, landing_cov = compute_run_statistics(results)

    return jsonify(dict(
        landing=dict(
            mean=landing_mean.tolist(),
            cov=landing_cov.tolist(),
        ),
    ))

# Mapping between prediction output formats and OGR drivers.
PREDICT_FORMAT_MAP = {
    'geojson': ('GeoJSON', 'json'), # NB. GeoJSON is natively emitted.
    'kml': ('LibKML', 'kml'),
}

@api.route('/predict.<format>', methods=['POST'])
def predict_reformat(format):
    if format not in PREDICT_FORMAT_MAP:
        raise NotFound()

    # Parse request body
    prediction_spec = parse_prediction_spec_request()

    # Modal tracks
    modal_tracks = run_modal_prediction(prediction_spec)

    # Run predictions
    results = run_predictions(prediction_spec)

    # Compute some stats
    l_mean, l_cov = compute_run_statistics(results)

    geojson_fc = runs_to_geojson(modal_tracks, l_mean, l_cov)

    if format == 'geojson':
        # We're done
        return jsonify(geojson_fc)

    # Otherwise, convert via OGR
    return ogr_convert_geojson(geojson_fc, *PREDICT_FORMAT_MAP[format])

LaunchSpec = namedtuple('LaunchSpec', ['lng', 'lat', 'alt', 'when'])
SimpleAltitudeProfile = namedtuple(
    'SimpleAltitudeProfile',
    ['ascent_rate', 'burst_alt', 'descent_rate']
)
PredictionSpec = namedtuple(
    'PredictionSpec',
    ['launch', 'profile', 'sample_count']
)

### FORMATTING OUTPUT ###

@contextmanager
def vsi_open(path, mode):
    """A context manager for GDAL's VSI filesystem interface which ensures that
    files are closed after use.

    """
    handle = gdal.VSIFOpenL(path, mode)
    yield handle
    gdal.VSIFCloseL(handle)

@contextmanager
def vsi_tempfile(prefix, extension):
    """A context manage which creates a temporary VSI file, returns the name
    and then unlinks it. Note that this is *not* safe against malicious intent
    and so one should only use this for /vsimem/... paths.

    """
    alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    temp_id = ''.join(list(random.choice(alpha) for _ in range(6)))
    name = '{0}{1}.{2}'.format(prefix, temp_id, extension)
    hn = gdal.VSIFOpenL(name, 'w')
    gdal.VSIFCloseL(hn)
    yield name
    rv = gdal.Unlink(name)
    assert rv == 0

def ogr_convert_geojson(src, driver_name, extension):
    in_tmp = vsi_tempfile('/vsimem/input', 'geojson')
    out_tmp = vsi_tempfile('/vsimem/output', extension)

    with in_tmp as in_path, out_tmp as out_path:
        gdal.FileFromMemBuffer(in_path, json.dumps(src))
        src_ds = ogr.Open(in_path)
        assert src_ds is not None

        drv = ogr.GetDriverByName(driver_name)
        assert drv is not None

        dst_ds = drv.CreateDataSource(out_path)
        assert dst_ds is not None

        for l_idx in range(src_ds.GetLayerCount()):
            layer = src_ds.GetLayerByIndex(l_idx)
            dst_ds.CopyLayer(layer, layer.GetName())

        dst_ds.SyncToDisk()

        with vsi_open(out_path, 'rb') as dst_handle:
            assert dst_handle is not None

            # How much data is there
            gdal.VSIFSeekL(dst_handle, 0, 2)
            total_size = gdal.VSIFTellL(dst_handle)
            gdal.VSIFSeekL(dst_handle, 0, 0)

            dst_data = gdal.VSIFReadL(1, total_size, dst_handle)
            assert dst_data is not None

    return dst_data

def runs_to_geojson(modal_tracks, l_mean, l_cov):
    modal_tracks_wgs84 = list(geocentric_to_wgs84(t) for t in modal_tracks)

    # List to hold output features
    features = []

    # Create modal track feature
    features.append(dict(
        type='Feature',
        geometry=dict(
            type='MultiLineString',
            coordinates=list(t[:, 1:].tolist() for t in modal_tracks_wgs84),
        ),
        properties=dict(
            times=list(t[:, 0].tolist() for t in modal_tracks_wgs84),
            OGR_STYLE='PEN(c:#00FF00)',
            altitudeMode='absolute',
        ),
    ))

    # Compute landing covariance 1 sigma vectors (ignoring time)
    lambdas, sigma_vs = np.linalg.eig(l_cov[1:, 1:])
    for col_idx in range(3):
        sigma_vs[:, col_idx] *= np.sqrt(np.maximum(0, lambdas[col_idx]))

    # Columns of sigma_vs are lat, lng, alt triples pointing along principal
    # axes of ellipse of uncertainty. Choose those with *largest* uncertainty.
    sigma_sq_mags = np.sum(sigma_vs ** 2, axis=0)
    ellipse_axes = sigma_vs[:, np.argsort(sigma_sq_mags)[-2:]]

    # Generate ellipse
    n_thetas = 32
    sigma_fills = ['#FF0000FF', '#FF0000BB', '#FF000088', '#FF000044']
    for sigma_idx, fill in enumerate(sigma_fills):
        sigma = sigma_idx + 1

        thetas = np.linspace(0, 2*np.pi, n_thetas)
        sins, coss = np.sin(thetas), np.cos(thetas)
        landing_poly_coords = np.repeat(
            l_mean[1:].reshape((1, -1)), n_thetas, axis=0)
        landing_poly_coords += np.dot(
            sins.reshape((-1, 1)),
            sigma * ellipse_axes[:, 0].reshape((1, -1))
        )
        landing_poly_coords += np.dot(
            coss.reshape((-1, 1)),
            sigma * ellipse_axes[:, 1].reshape((1, -1))
        )

        # Convert polygon co-ordinates to WGS84 lng/lats. Notice that we ignore
        # altitude.
        lp_coords_wgs84 = np.zeros((n_thetas, 2))
        lp_coords_wgs84[:, 0], lp_coords_wgs84[:, 1], _ = pyproj.transform(
            GEOCENT_PROJECTION, WGS84_PROJECTION,
            landing_poly_coords[:, 0], landing_poly_coords[:, 1],
            landing_poly_coords[:, 2]
        )

        features.append(dict(
            type='Feature',
            geometry=dict(
                type='Polygon',
                coordinates=[lp_coords_wgs84.tolist()],
            ),
            properties=dict(
                drawOrder=len(sigma_fills) - sigma_idx,
                OGR_STYLE='BRUSH(fc:{0});PEN(c:#000000)'.format(fill),
            )
        ))

    return dict(type='FeatureCollection', features=features)

### RUNNING THE PREDICTOR ###

def ruaumoko_ds():
    if not hasattr("ruaumoko_ds", "once"):
        ds_loc = current_app.config.get(
            'ELEVATION_DATASET', ElevationDataset.default_location
        )
        ruaumoko_ds.once = ElevationDataset(ds_loc)

    return ruaumoko_ds.once

WGS84_PROJECTION = pyproj.Proj(init='epsg:4326')
GEOCENT_PROJECTION = pyproj.Proj(proj='geocent', datum='WGS84', units='m')

def solver_to_geocentric(observations):
    """Take a Nx4 array of time, latitude, longitude, altitude observations and
    convert it to a Nx4 array of time, x, y and z geo-centric co-ordinates.
    Time is measured in seconds since the epoch and x, y and z are measured in
    metres.

    """
    output = observations.copy()
    output[:, 1], output[:, 2], output[:, 3] = pyproj.transform(
        WGS84_PROJECTION, GEOCENT_PROJECTION,
        observations[:, 2], observations[:, 1], observations[:, 3]
    )
    return output

def geocentric_to_wgs84(observations):
    """Take a Nx4 array of time, x, y, z observations and convert it to a Nx4
    array of time, longitude, latitude and altitude geo-centric co-ordinates.
    Time is measured in seconds since the epoch and x, y and z are measured in
    metres. NOTE ordering of longitude and latitude. This function is *NOT* the
    inverse of solver_to_geocentric().

    """
    output = observations.copy()
    output[:, 1], output[:, 2], output[:, 3] = pyproj.transform(
        GEOCENT_PROJECTION, WGS84_PROJECTION,
        observations[:, 1], observations[:, 2], observations[:, 3]
    )
    return output

def run_modal_prediction(spec):
    """Sample and run a modal prediction according to the passed PredictionSpec.

    Returns a Nx4 array where each row is a prediction for a different time
    point. Each prediction is a 4-tuple of UNIX timestamp, latitude, longitude,
    altitude.

    """
    # Find wind data location
    ds_dir = current_app.config.get(
        'WIND_DATASET_DIR', WindDataset.DEFAULT_DIRECTORY
    )

    # Load dataset
    tawhiri_ds = WindDataset.open_latest(persistent=True, directory=ds_dir)

    # Ascent profile
    ascent_rate = spec.profile.ascent_rate.mode
    descent_rate = spec.profile.descent_rate.mode
    burst_alt = spec.profile.burst_alt.mode

    # Create ascent profile
    stages = models.standard_profile(
        ascent_rate, burst_alt, descent_rate,
        tawhiri_ds, ruaumoko_ds()
    )

    # Launch site
    lat = spec.launch.lat.mode
    lng = spec.launch.lng.mode
    alt = spec.launch.alt.mode
    when = spec.launch.when.mode

    return list(
        solver_to_geocentric(np.array(r))
        for r in solver.solve(when, lat, lng, alt, stages)
    )

def run_predictions(spec):
    """Sample and run predictions according to the passed PredictionSpec.

    Returns a sequence of sequences of a Nx4 arrays, one per leg, where each
    row is a prediction for a different time point. Each prediction is a
    4-tuple of UNIX timestamp, latitude, longitude, altitude.

    """
    # Find wind data location
    ds_dir = current_app.config.get(
        'WIND_DATASET_DIR', WindDataset.DEFAULT_DIRECTORY
    )

    # Load dataset
    tawhiri_ds = WindDataset.open_latest(persistent=True, directory=ds_dir)

    # TODO: make this Python2 friendly
    results = []
    for _ in range(spec.sample_count):
        # Draw ascent profile
        ascent_rate = spec.profile.ascent_rate()
        descent_rate = spec.profile.descent_rate()
        burst_alt = spec.profile.burst_alt()

        # Create ascent profile
        stages = models.standard_profile(
            ascent_rate, burst_alt, descent_rate,
            tawhiri_ds, ruaumoko_ds()
        )

        # Draw launch site
        lat = spec.launch.lat()
        lng = spec.launch.lng()
        alt = spec.launch.alt()
        when = spec.launch.when()

        data = list(
            solver_to_geocentric(np.array(r))
            for r in solver.solve(when, lat, lng, alt, stages)
        )

        # Add to data list
        results.append(data)

    return results

def compute_run_statistics(results):
    """Compute statistics from a sequence of runs."""

    # Compute landing points, sum and sample count. When computing running sum,
    # convert time axis to *relative* time.
    landing_points = np.zeros((len(results), 4))
    for r_idx, res in enumerate(results):
        landing_points[r_idx, :] = res[-1][-1, :]

    # Compute mean and covariance of landing
    landing_mean = np.mean(landing_points, axis=0)
    landing_cov = np.cov(landing_points.T)

    return landing_mean, landing_cov

### PARSING OF SPEC FROM JSON REQUESTS ###

def parse_prediction_spec_request():
    # Parse request body as JSON. Note that get_json() will return None if the
    # request body is empty.
    json = request.get_json()
    if json is None:
        raise BadRequest()

    # Parse JSON body into a prediction spec.
    try:
        prediction_spec = prediction_spec_from_json(json)
    except (IndexError, ValueError, KeyError) as e:
        if current_app.debug:
            raise e
        else:
            raise BadRequest()

    return prediction_spec

def prediction_spec_from_json(json):
    """Construct a new PredictionSpec from a dictionary representing a
    prediction specification encoded as per the API documentation. Raises
    ValueError if the dictionary is in some way invalid.

    """
    launch = launch_from_json(json['launch'])
    profile = profile_from_json(json['profile'])
    sample_count = int(json['sampleCount'])

    if sample_count <= 0:
        raise ValueError('Sample count must be positive')
    if sample_count > 1000:
        raise ValueError('Sample count is too large')

    return PredictionSpec(
        launch=launch, profile=profile, sample_count=sample_count
    )

def launch_from_json(json):
    """Construct a new LaunchSpec from a dictionary representing a launch
    specification.

    """
    return LaunchSpec(
        lat=sampleable_from_json(json['latitude']),
        lng=sampleable_from_json(json['longitude']),
        alt=sampleable_from_json(json['altitude']),
        when=sampleable_from_json(json['when']),
    )

def profile_from_json(json):
    """Construct a new SimpleAltitudeProfile from a dictionary representing an
    altitude profile specification.

    """
    type_ = json['type']
    if type_ == 'simple':
        return SimpleAltitudeProfile(
            ascent_rate=sampleable_from_json(json['ascentRate']),
            descent_rate=sampleable_from_json(json['descentRate']),
            burst_alt=sampleable_from_json(json['burstAltitude']),
        )

    raise ValueError('Unknown altitude profile type: ' + str(type_))

def sampleable_from_json(json):
    """Construct a sampleable from the passed dict representing a json
    object or number.

    A sampleable is simply a callable which returns a floating point value
    drawn from some distribution. It has a mode attribute giving the mode of
    the distribution.

    """
    try:
        # Treat as simple number
        num = float(json)
        sampleable = lambda: num
        sampleable.mode = num
        return sampleable
    except (TypeError, ValueError):
        # OK, continue trying to treat json as a dict
        pass

    type_ = json['type']
    if type_ == 'gaussian':
        mu, sigma = float(json['mu']), float(json['sigma'])
        sampleable = lambda: random.gauss(mu, sigma)
        sampleable.mode = mu
        return sampleable
    elif type_ == 'uniform':
        low, high = float(json['low']), float(json['high'])
        sampleable = lambda: random.uniform(low, high)
        sampleable.mode = 0.5 * (low + high)
        return sampleable
    else:
        raise ValueError('Bad sampleable type: ' + repr(type_))
