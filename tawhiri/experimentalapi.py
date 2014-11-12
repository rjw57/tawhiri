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
import random

from flask import Blueprint, jsonify, request, current_app
import numpy as np
from werkzeug.exceptions import BadRequest

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
    mean_track, landing_mean, landing_cov = compute_run_statistics(results)

    return jsonify(dict(
        mean=mean_track.tolist(),
        landing=dict(
            mean=landing_mean.tolist(),
            cov=landing_cov.tolist(),
        ),
    ))

@api.route('/predict.geojson', methods=['POST'])
def predict_geojson():
    # Parse request body
    prediction_spec = parse_prediction_spec_request()

    # Modal track
    modal_track = run_modal_prediction(prediction_spec)

    # Run predictions
    results = run_predictions(prediction_spec)

    # Compute some stats
    _, l_mean, l_cov = compute_run_statistics(results)

    # List to hold output features
    features = []

    # Create modal track feature
    features.append(dict(
        type='Feature',
        geometry=dict(
            type='LineString',
            # Note swizzling of lat/lng into lng/lat order
            coordinates=modal_track[:, [2, 1, 3]].tolist()
        ),
        properties=dict(
            times=modal_track[:, 0].tolist(),
            OGR_STYLE='PEN(c:#00FF00)',
            altitudeMode='absolute',
        ),
    ))

    # Compute landing covariance 1 sigma vectors (ignoring time)
    lambdas, sigma_vs = np.linalg.eig(l_cov[1:, 1:])
    for col_idx in range(3):
        sigma_vs[:, col_idx] *= np.sqrt(np.maximum(0, lambdas[col_idx]))

    # Columns of sigma_vs are lat, lng, alt triples pointing along principal
    # axes of ellipse of uncertainty. Choose those with *smallest* altitudes.
    ellipse_axes = sigma_vs[:, np.argsort(sigma_vs[2, :])[:2]]

    # Generate ellipse
    n_thetas = 32
    thetas = np.linspace(0, 2*np.pi, n_thetas)
    sins, coss = np.sin(thetas), np.cos(thetas)
    landing_poly_coords = np.repeat(
        l_mean[1:3].reshape((1, -1)), n_thetas, axis=0)
    landing_poly_coords += np.dot(
        sins.reshape((-1, 1)),
        3 * ellipse_axes[:2, 0].reshape((1, -1))
    )
    landing_poly_coords += np.dot(
        coss.reshape((-1, 1)),
        3 * ellipse_axes[:2, 1].reshape((1, -1))
    )

    features.append(dict(
        type='Feature',
        geometry=dict(
            type='Polygon',
            # Note swizzling of lat/lng into lng/lat order
            coordinates=[landing_poly_coords[:, [1, 0]].tolist()],
        ),
        properties=dict(
            OGR_STYLE='BRUSH(fc:#FF000080);PEN(c:#FFFFFF00)',
        )
    ))

    return jsonify(dict(
        type='FeatureCollection',
        features=features
    ))

LaunchSpec = namedtuple('LaunchSpec', ['lng', 'lat', 'alt', 'when'])
SimpleAltitudeProfile = namedtuple(
    'SimpleAltitudeProfile',
    ['ascent_rate', 'burst_alt', 'descent_rate']
)
PredictionSpec = namedtuple(
    'PredictionSpec',
    ['launch', 'profile', 'sample_count']
)

### RUNNING THE PREDICTOR ###

def ruaumoko_ds():
    if not hasattr("ruaumoko_ds", "once"):
        ds_loc = current_app.config.get(
            'ELEVATION_DATASET', ElevationDataset.default_location
        )
        ruaumoko_ds.once = ElevationDataset(ds_loc)

    return ruaumoko_ds.once


def run_modal_prediction(spec):
    """Sample and run a modal prediction according to the passed PredictionSpec.

    Returns a sequence of a Nx4 arrays where each row is a prediction for a
    different time point.  Each prediction is a 4-tuple of UNIX timestamp,
    latitude, longitude, altitude.

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

    # Concatenate legs together into single Nx4 array
    return np.vstack(solver.solve(when, lat, lng, alt, stages))

def run_predictions(spec):
    """Sample and run predictions according to the passed PredictionSpec.

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

        # Concatenate legs together into single Nx4 array
        data = np.vstack(solver.solve(when, lat, lng, alt, stages))

        # Add to data list
        results.append(data)

    return results

def compute_run_statistics(results):
    """Compute statistics from a sequence of runs."""

    # Compute landing points, sum and sample count. When computing running sum,
    # convert time axis to *relative* time.
    max_len = np.max(list(r.shape[0] for r in results))
    landing_points = np.zeros((len(results), 4))
    sample_sum = np.zeros((max_len, 4))
    sample_count = np.zeros((max_len,), np.int)
    for r_idx, res in enumerate(results):
        landing_points[r_idx, :] = res[-1, :]
        sample_sum[:res.shape[0], :] += res
        sample_sum[:res.shape[0], 0] -= res[0, 0]
        sample_count[:res.shape[0]] += 1

    # Compute mean track
    mean_track = sample_sum.copy()
    mean_track /= np.repeat(sample_count.reshape((-1, 1)), 4, axis=1)

    # Compute mean and covariance of landing
    landing_mean = np.mean(landing_points, axis=0)
    landing_cov = np.cov(landing_points.T)

    return mean_track, landing_mean, landing_cov

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
