"""
Utility functions for creating output in a variety of geospatial file formats.

"""
from contextlib import contextmanager
import random

import pyproj
import numpy as np
from osgeo import ogr, osr, gdal

from .util import GEOCENT_PROJECTION, WGS84_PROJECTION

_WGS84_OGR_SRS = osr.SpatialReference(osr.SRS_WKT_WGS84)

@contextmanager
def new_vector(out_fobj, extension, driver_name):
    """A context manager which returns an OGR dataset which may have layers
    appended. After leaving the context, the dataset is written in the selected
    format to out_fobj. The data source is created as an in-memory file with
    the specified extension and using the specified driver.

    """
    with _vsi_tempfile('/vsimem/output', extension) as out_fn:
        drv = ogr.GetDriverByName(driver_name)
        if drv is None:
            raise ValueError('Error creating OGR driver: ' + str(driver_name))
        dst_ds = drv.CreateDataSource(out_fn)
        if dst_ds is None:
            raise ValueError('Error creating DataSouce')

        yield dst_ds

        # Release dataset
        dst_ds.Destroy()

        # Read out body
        out_fobj.write(_vsi_read(out_fn))

def append_tracks_layer(ds, legs, layer_name='legs'):
    """Create a layer in ds with features representing legs as a LineString.

    Legs must be a sequence of Nx4 arrays with rows being time, longitude,
    latitude and altitude relative to the _WGS84 datum.

    """
    # Create layer
    layer = ds.CreateLayer(layer_name, geom_type=ogr.wkbLineString)
    assert layer is not None

    # Create fields
    layer.CreateField(ogr.FieldDefn('altitudeMode', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('times', ogr.OFTRealList))

    for leg in legs:
        # Create leg geometry
        leg_geom = ogr.Geometry(ogr.wkbLineString)
        assert leg_geom is not None
        leg_geom.AssignSpatialReference(_WGS84_OGR_SRS)

        for _, lng, lat, alt in leg:
            leg_geom.AddPoint(lng, lat, alt)

        # Create leg feature
        feature = ogr.Feature(layer.GetLayerDefn())
        assert feature is not None
        feature.SetGeometry(leg_geom)
        feature.SetStyleString('PEN(c:#00FF00)')
        feature.SetField('altitudeMode', 'absolute')
        feature.SetFieldDoubleList(
            feature.GetFieldIndex('times'), leg[:, 0].tolist()
        )
        layer.CreateFeature(feature)

    return feature

def append_2d_covariance_layer(ds, mean, covariance, layer_name='uncertainty'):
    """Create a layer in ds with features representing the 4x4 covariance
    matrix cov centres on mean. The covariance and mean dimensions are assumed
    to be ordered time, x, y, z. (I.e. in geocrentric co-ordinates.).

    The covariance axes with maximum uncertainty are chosen and projected onto
    the ground. 1, 2, 3 and 4 sigma ellipses are then drawn.

    """
    # Create layer
    layer = ds.CreateLayer(layer_name, geom_type=ogr.wkbPolygon)
    assert layer is not None

    # Create fields
    layer.CreateField(ogr.FieldDefn('altitudeMode', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('drawOrder', ogr.OFTInteger))

    # Compute covariance 1 sigma vectors (ignoring time)
    sigma_vs = _cov_one_sigma_axes(covariance[1:, 1:])

    # Columns of sigma_vs are x, y, z triples pointing along principal
    # axes of ellipse of uncertainty. Choose those with *largest* uncertainty.
    sigma_sq_mags = np.sum(sigma_vs ** 2, axis=0)
    ellipse_axes = sigma_vs[:, np.argsort(sigma_sq_mags)[-2:]]

    # Generate ellipse
    sigma_fills = ['#FF0000FF', '#FF0000BB', '#FF000088', '#FF000044']
    for sigma_idx, fill in enumerate(sigma_fills):
        sigma = sigma_idx + 1
        poly_coords = points_on_ellipse(mean[1:], sigma * ellipse_axes)

        # Convert polygon co-ordinates to _WGS84 lng/lats. Notice that we ignore
        # altitude.
        coords_wgs84 = np.vstack(pyproj.transform(
            GEOCENT_PROJECTION, WGS84_PROJECTION,
            poly_coords[:, 0], poly_coords[:, 1], poly_coords[:, 2]
        )[:2]).T
        assert coords_wgs84.shape[1] == 2

        # Create polygon feature
        feature = _create_simple_polygon(layer.GetLayerDefn(), coords_wgs84)
        feature.SetStyleString('PEN(c:#000000);BRUSH(fc:{0})'.format(fill))
        feature.SetField('drawOrder', len(sigma_fills) - sigma_idx)
        layer.CreateFeature(feature)

    return feature

def _create_simple_polygon(field_defn, coords, srs=_WGS84_OGR_SRS):
    """Create and return an OGR feature representing a polygon with a single
    external ring. If srs is not None, it is a ogs.SptialReference which should
    be assigned to the feature.

    *coords* is be a Nx2 array of co-ordinates for the polygon.

    *field_defn* is the feature's field definition.

    """
    # Create geometry
    geom = ogr.Geometry(ogr.wkbPolygon)
    ring_geom = ogr.Geometry(ogr.wkbLinearRing)
    if srs is not None:
        geom.AssignSpatialReference(srs)
        ring_geom.AssignSpatialReference(srs)

    # Set points
    for lng, lat in coords[:, :2]:
        ring_geom.AddPoint(lng, lat)
    geom.AddGeometry(ring_geom)

    # Create feature
    feature = ogr.Feature(field_defn)
    feature.SetGeometry(geom)
    feature.SetField('altitudeMode', 'absolute')

    return feature

def points_on_ellipse(centre, axes, point_count=32):
    """Return point_count points around the ellpse centres on centre with the
    specified axes. The axes must be specified as a Nx2 array. The centre point
    should be a one-dimensional array of length N.

    The points are returned as a point_count x N array.

    """
    # Check arguments
    if len(axes.shape) != 2 or axes.shape[1] != 2:
        raise ValueError('Exactly two axes must be specified')

    # Calculate sin(theta), cos(theta) for each point
    thetas = np.linspace(0, 2*np.pi, point_count)
    sins, coss = np.sin(thetas), np.cos(thetas)

    # Create output co-ords array
    coords = np.repeat(centre.reshape((1, -1)), point_count, axis=0)
    coords += np.dot(sins.reshape((-1, 1)), axes[:, 0].reshape((1, -1)))
    coords += np.dot(coss.reshape((-1, 1)), axes[:, 1].reshape((1, -1)))

    return coords

### GDAL'S VSI LAYER ###

def _vsi_read(path):
    """Use GDAL's VSI routines to read the contents of the file at path."""
    # Read out body
    with _vsi_open(path, 'rb') as file_handle:
        assert file_handle is not None

        # How much data is there?
        gdal.VSIFSeekL(file_handle, 0, 2)
        total_size = gdal.VSIFTellL(file_handle)
        gdal.VSIFSeekL(file_handle, 0, 0)

        file_data = gdal.VSIFReadL(1, total_size, file_handle)
        assert file_data is not None
    return file_data

@contextmanager
def _vsi_open(path, mode):
    """A context manager for GDAL's VSI filesystem interface which ensures that
    files are closed after use.

    """
    handle = gdal.VSIFOpenL(path, mode)
    yield handle
    gdal.VSIFCloseL(handle)

@contextmanager
def _vsi_tempfile(prefix, extension):
    """A context manage which creates a temporary VSI file, returns the name
    and then unlinks it. Note that this is *not* safe against malicious intent
    and so one should only use this for /vsimem/... paths.

    """
    alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    temp_id = ''.join(list(random.choice(alpha) for _ in range(6)))
    name = '{0}{1}.{2}'.format(prefix, temp_id, extension)

    # Create, close and unlink the file to make sure it doesn't exist.
    hn = gdal.VSIFOpenL(name, 'w')
    gdal.VSIFCloseL(hn)
    gdal.Unlink(name)

    yield name

    # Now unlink the file if it exists
    gdal.Unlink(name)

### GEOMETRIC INTERPRETATION OF STATISTICAL MEASURES ###

def _cov_one_sigma_axes(cov):
    """Given a NxN array representing a Gaussian covariance return a NxN array
    whose columns are the principal axes of the one-sigma ellipse. There is no
    guarantee on the ordering of these axes.

    """
    if len(cov.shape) != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError('covariance must be square')

    lambdas, sigma_vs = np.linalg.eig(cov)
    for col_idx in range(cov.shape[0]):
        sigma_vs[:, col_idx] *= np.sqrt(np.maximum(0, lambdas[col_idx]))

    return sigma_vs
