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
from io import BytesIO
import random

from flask import Blueprint, jsonify, request, current_app
import numpy as np
from six.moves import xrange as range # pylint: disable=import-error,redefined-builtin
from werkzeug.exceptions import BadRequest, NotFound

from tawhiri import solver, models
from tawhiri.dataset import Dataset as WindDataset
from ruaumoko import Dataset as ElevationDataset

from .geoformats import new_vector
from .geoformats import append_tracks_layer, append_2d_covariance_layer
from .util import geocentric_to_wgs84, solver_to_geocentric

api = Blueprint('api_experimental', __name__)

# Mapping between prediction output formats and OGR drivers, extensions and
# media types.
PREDICT_FORMAT_MAP = {
    'kmz': ('LibKML', 'kmz', 'application/vnd.google-earth.kmz'),
    'kml': ('LibKML', 'kml', 'application/vnd.google-earth.kml+xml'),
}

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

@api.route('/predict.<ext>', methods=['POST'])
def predict_reformat(ext):
    # Which OGR driver should we use for this?
    try:
        driver_name, extension, media_type = PREDICT_FORMAT_MAP[ext]
    except TypeError as e:
        # If we're running in debug mode, raise the original exception
        # otherwise, treat as if this endpoint doesn't exit.
        if current_app.debug:
            raise e
        raise NotFound()

    # Parse request body
    prediction_spec = parse_prediction_spec_request()

    # Run predictions
    modal_tracks = run_modal_prediction(prediction_spec)
    results = run_predictions(prediction_spec)

    # Compute some stats
    l_mean, l_cov = compute_run_statistics(results)

    # Transform into output SRS
    modal_tracks_wgs84 = list(geocentric_to_wgs84(t) for t in modal_tracks)

    # Create output file and OGR data source
    data = BytesIO()
    with new_vector(data, extension, driver_name) as dst_ds:
        # Create output features
        append_tracks_layer(dst_ds, modal_tracks_wgs84)
        append_2d_covariance_layer(dst_ds, l_mean, l_cov)

    return (data.getvalue(), 200, {'Content-Type': media_type})

### RUNNING THE PREDICTOR ###

LaunchSpec = namedtuple('LaunchSpec', ['lng', 'lat', 'alt', 'when'])
SimpleAltitudeProfile = namedtuple(
    'SimpleAltitudeProfile',
    ['ascent_rate', 'burst_alt', 'descent_rate']
)
PredictionSpec = namedtuple(
    'PredictionSpec',
    ['launch', 'profile', 'sample_count']
)

def ruaumoko_ds():
    if not hasattr("ruaumoko_ds", "once"):
        ds_loc = current_app.config.get(
            'ELEVATION_DATASET', ElevationDataset.default_location
        )
        ruaumoko_ds.once = ElevationDataset(ds_loc)

    return ruaumoko_ds.once

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
