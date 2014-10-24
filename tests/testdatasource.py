from logging import getLogger
from nose.tools import ok_, raises

import tawhiri.datasource as tds

LOG = getLogger(__name__)

def test_index():
    ds = tds.DatasetIndex(refresh_immediately=False)

    LOG.info('Initial dataset index: {0}'.format(ds.index))
    ok_(len(ds.index) == 0, 'Initial dataset index is empty')
    ok_(ds.latest_run is None, 'No latest run')

    LOG.info('Refreshing...')
    ds.refresh(limit_runs=2)

    ok_(ds.latest_run is not None, 'After refresh, there is a latest run')
    ok_(ds.index is not None, 'After refresh, dataset index is non-none')
    LOG.info('New dataset index has {0} item(s)'.format(len(ds.index)))
    ok_(len(ds.index) > 0, 'After refresh, dataset index has some items')
    ok_(len(ds.index) <= 2, 'After refresh, dataset index has at most two items')

def test_latest_gribs():
    ds = tds.DatasetIndex(refresh_immediately=True)
    ok_(len(ds.latest_gribs_of_type('f')) > 0, 'Found some "f" GRIBs')
    ok_(len(ds.latest_gribs_of_type('bf')) > 0, 'Found some "bf" GRIBs')

@raises(IndexError)
def test_latest_gribs_requires_refresh():
    ds = tds.DatasetIndex(refresh_immediately=False)
    lgs = ds.latest_gribs_of_type('f') # should raise
