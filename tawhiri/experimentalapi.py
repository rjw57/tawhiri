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

Overview
========

The primary entry point to the API is .../predict which should have a JSON body
POST-ed to it with the following form:

    <request> := {
        "launch": <launchSpec>,
        "profile": <altitudeProfile>,
        "sampleCount": <number>
    }

The sampleCount field provides an indication of the desired number of sample
trajectories drawn. It is a hint only. The predictor may decide to draw fewer
samples if the prediction is taking too long.

A launchSpec specifies where and when a launch is intended to occur and has the
following form:

    <launchSpec> := {
        "latitude": <sampleable>,
        "longitude": <sampleable>,
        "altitude": <sampleable>,
        "when": <sampleable>
    }

The latitude and longitude fields are specified in degrees. The altitude field
is specified in metres. Together they locate the launch site in space. The
when field is specified as the number of seconds since UTC 00:00 01 Jan 1970.
The value may be negative.

An altitudeProfile specifies the expected altitude profile for the balloon and
has the following form::

    <altitudeProfile> := {
        "type": "simple",
        "parameters": <simpleProfileParameters>
    }

Currently the only supported type is "simple". The parameters field contains
profile-type dependent fields. For the "simple" type the parameters object
should have the following format::

    <simpleProfileParameters> := {
        "ascentRate": <sampleable>,
        "descentRate": <sampleable>,
        "burstAltitude": <sampleable>,
    }

A sampleable is defined as follows::

    <sampleable> := <number>
                  | <gaussianDistribution>
                  | <uniformDistribution>

It represents some distribution which is sampled from to obtain a particular
value. If it is a number then the distribution is assumed to be a
delta-function lying at that value.

Other distributions are defined as follows::

    <gaussianDistribution> := {
        "type": "gaussian",
        "mu": <number>,             // mean value
        "sigma": <number>           // standard deviation
    }

    <uniformDistribution> := {
        "type": "uniform",
        "low": <number>,            // samples will be >= low
        "high": <number>,           // samples will be <= high
    }

Other entry points
==================

Requests to .../ will return the following JSON object::

    {
        "version": 2
    }

"""
from collections import namedtuple
import random

from flask import Blueprint, jsonify, request, current_app
from werkzeug.exceptions import BadRequest

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

    try:
        prediction_spec = prediction_spec_from_json(json)
    except (IndexError, ValueError, KeyError) as e:
        if current_app.debug:
            raise e
        else:
            raise BadRequest()

    return str(prediction_spec)

LaunchSpec = namedtuple('LaunchSpec', ['lng', 'lat', 'alt', 'when'])
SimpleAltitudeProfile = namedtuple(
    'SimpleAltitudeProfile',
    ['ascent_rate', 'burst_alt', 'descent_rate']
)
PredictionSpec = namedtuple(
    'PredictionSpec',
    ['launch', 'profile', 'sample_count']
)

def prediction_spec_from_json(json):
    """Construct a new PredictionSpec from a dictionary representing a
    prediction specification encoded as per the API documentation. Raises
    ValueError if the dictionary is in some way invalid.

    """
    launch = launch_from_json(json['launch'])
    profile = profile_from_json(json['profile'])
    sample_count = float(json['sampleCount'])
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
    type_, params = json['type'], json['parameters']
    if type_ == 'simple':
        return SimpleAltitudeProfile(
            ascent_rate=sampleable_from_json(params['ascentRate']),
            descent_rate=sampleable_from_json(params['descentRate']),
            burst_alt=sampleable_from_json(params['burstAltitude']),
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
