"""
Run a tawhiri landing prediction.

"""
from collections import namedtuple
from flask import current_app

from tawhiri import solver
from tawhiri.dataset import Dataset as WindDataset
from ruaumoko import Dataset as ElevationDataset

LaunchLocation = namedtuple(
    'LaunchLocation', ['latitude', 'longitude', 'altitude', 'when']
)
LaunchLocation.__doc__ += '''
A convenience namedtuple for passing a location for launch.
'''

PredictionRun = namedtuple(
    'PredictionRun', ['dataset_datetime', 'solution']
)

class NoSuchDatasetException(RuntimeError):
    pass

def run_prediction(whence, altitude_profile, dataset_datetime=None):
    """Run a tawhiri landing prediction. *whence* is a latitude (deg),
    longitude (deg), altitude (m), time (datetime) 4-tuple specifying when the
    prediction should be run from. *altitude_profile* is an altitude profile
    (e.g. as returned by tawhiri.models.standard_profile).

    The altitude in *whence* may be None in which case ruaumoko is used to
    query the appropriate ground-surface altitude for that location.

    If dataset_datetime is None, use the latest available wind dataset.
    Otherwise it is a datetime instance specifying which dataset should be
    used.

    Raises NoSuchDatasetException if there is no dataset for the specified time.

    Returns a PredictionRun tuple.

    """
    # Find wind data location
    ds_dir = current_app.config.get(
        'WIND_DATASET_DIR', WindDataset.DEFAULT_DIRECTORY
    )

    # Dataset
    try:
        if dataset_datetime is None:
            tawhiri_ds = WindDataset.open_latest(
                persistent=True, directory=ds_dir
            )
        else:
            tawhiri_ds = WindDataset(dataset_datetime, directory=ds_dir)
    except IOError:
        raise NoSuchDatasetException("No matching dataset found.")
    except ValueError as e:
        raise NoSuchDatasetException(*e.args)

    # Extract launch location
    launch_lat, launch_lng, launch_alt, launch_dt = whence[:4]

    # Fixup altitude if required
    if launch_alt is None:
        launch_alt = _get_ruaumoko_ds().get(launch_lat, launch_lng)

    # Run solver
    solution = solver.solve(
        launch_dt, launch_lat, launch_lng, launch_alt,
        altitude_profile
    )

    return PredictionRun(tawhiri_ds.ds_time, solution)

def _get_ruaumoko_ds():
    """Load the Ruaumoko dataset. The return value is memoized so that repeated
    calls to this function are efficient.

    """
    if not hasattr("_get_ruaumoko_ds", "once"):
        ds_loc = current_app.config.get(
            'ELEVATION_DATASET', ElevationDataset.default_location
        )
        _get_ruaumoko_ds.once = ElevationDataset(ds_loc)

    return _get_ruaumoko_ds.once
