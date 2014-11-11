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

    # Run predictions
    means, ellps = run_predictions(prediction_spec)

    return jsonify(dict(means=means.tolist(), sigmavecs=ellps.tolist()))

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

def run_predictions(spec):
    """Sample and run predictions according to the passed PredictionSpec.

    """

    # Find wind data location
    ds_dir = current_app.config.get(
        'WIND_DATASET_DIR', WindDataset.DEFAULT_DIRECTORY
    )

    # Load dataset
    tawhiri_ds = WindDataset.open_latest(persistent=True, directory=ds_dir)

    # TODO: make this Python2 friendly
    results, max_shape = [], (0, 0)
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

        # Change time axis to be *relative* offsets
        data[:, 0] -= data[0, 0]

        max_shape = tuple(max(ms, s) for ms, s in zip(max_shape, data.shape))

        # Add to data list
        results.append(data)

    return compute_run_stats(results, max_shape)

def cov_to_sigma_ellipse(cov):
    """Return array whose columns represent one sigma vectors along the
    principal axes.

    """
    # Compute eigenvalues and eigenvectors
    lambdas, vs = np.linalg.eig(cov)

    # Compute "one sigma" vectors
    for idx, sigma in enumerate(np.sqrt(np.maximum(0, lambdas))):
        vs[:, idx] *= sigma
    vs = np.ma.getdata(np.ma.fix_invalid(vs, fill_value=0))

    # Choose most vertical (i.e. one with greatest abs z component)
    most_vert_idx = np.argmax(np.abs(vs[2, :]))
    vert_v = np.atleast_2d(vs[:, most_vert_idx]).T
    if vert_v[2, 0] < 0:
        vert_v *= -1
    vs = np.hstack((vs[:, :most_vert_idx], vs[:, most_vert_idx+1:]))

    # Choose most north-south
    most_ns_idx = np.argmax(np.abs(vs[1, :]))
    ns_v = np.atleast_2d(vs[:, most_ns_idx]).T
    if ns_v[1, 0] < 0:
        ns_v *= -1
    vs = np.hstack((vs[:, :most_ns_idx], vs[:, most_ns_idx+1:]))

    assert vs.shape[1] == 1
    if vs[0, 0] < 0:
        vs *= -1

    # Use this order
    vs = np.hstack((vert_v, ns_v, vs))

    return vs

def compute_run_stats(results, max_shape):
    # Form combined array of timesteps x obs x samples
    combined_runs = np.zeros(max_shape + (len(results),))
    combined_runs[:] = np.nan
    for idx, r in enumerate(results):
        combined_runs[:r.shape[0], :, idx] = r
    assert combined_runs.shape[1] == 4

    # Convert combined_runs into a masked array
    combined_runs = np.ma.masked_invalid(combined_runs)

    # Compute ensemble mean
    means = np.ma.mean(combined_runs, axis=-1)

    # Compute one sigma ellipses. Note that since we're dealing with relative
    # time, we expect the covariance of the time dimension to be negligible.
    ellps = list(
        cov_to_sigma_ellipse(np.ma.getdata(np.ma.cov(row[1:, :])))
        for row in combined_runs
    )

    # Output is means (Nx4), ellps (Nx4x4). Each "row" of ellps being an array
    # whose rows are the one-sigma vectors.
    return np.ma.getdata(means), np.dstack(ellps).T

### PARSING OF SPEC FROM JSON REQUESTS ###

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
    drawn from some distribution.

    """
    try:
        # Treat as simple number
        num = float(json)
        return lambda: num
    except (TypeError, ValueError):
        # OK, continue trying to treat json as a dict
        pass

    type_ = json['type']
    if type_ == 'gaussian':
        mu, sigma = float(json['mu']), float(json['sigma'])
        return lambda: random.gauss(mu, sigma)
    elif type_ == 'uniform':
        low, high = float(json['low']), float(json['high'])
        return lambda: random.uniform(low, high)
    else:
        raise ValueError('Bad sampleable type: ' + repr(type_))
