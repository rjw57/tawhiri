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
#
"""Data provider for NCEP/NOAA GFS data.
"""
from collections import namedtuple
import itertools
import logging
import re

import requests

LOG = logging.getLogger(__name__)

#: Root URL for datasets
DATA_ROOT = 'http://ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/'

#: Pattern used to match GFS datasets
GFS_RUN_PATTERN = re.compile(
    'gfs.(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})(?P<hour>[0-9]{2})')

#: Pattern used to match individual GRIB files in datasets
GRIB_PATTERN = re.compile('gfs.t(?P<runhour>[0-9]{2})z.pgrb2(?P<formid>[a-z]+)(?P<hour>[0-9]+)')

class RunRecord(namedtuple('RunRecord', ('year', 'month', 'day', 'hour'))):
    """A description of an individual run of the GFS parameterised by the
    integer attributes *year*, *month*, *day* and *hour*. These may also be
    accessed using the indices 0 to 3.

    The *url* attribute has the path to the run's data on the GFS data server
    and the *id* attribute has a string identifying the run.

    """
    @classmethod
    def from_match(cls, match):
        kwargs = dict((key, int(match.group(key))) for key in cls._fields)
        return cls(**kwargs)

    @property
    def id(self):
        return 'run{0.year:04d}{0.month:02d}{0.day:02d}{0.hour:02d}'.format(self)

    @property
    def url(self):
        return DATA_ROOT + 'gfs.{0.year:04d}{0.month:02d}{0.day:02d}{0.hour:02d}'.format(self) + '/'

class GribRecord(namedtuple('GribRecord', ('run', 'hour', 'formid'))):
    """A description of an individual GRIB from an individual run parameterised
    by *run*, a RunRecord instance, *formid* which is a code describing the
    contents of the file and *hour* which is an integer giving the hour into
    the future the GRIB represents.

    The *url* attribute has the path to the GRIB's file on the GFS data server
    and the *id* attribute has a string identifying the GRIB.

    """
    @classmethod
    def from_match(cls, run, match):
        runhour = int(match.group('runhour'))
        if runhour != run.hour:
            raise ValueError('Matched hour {0} does not correspond to run {1.id}'.format(
                runhour, run))
        kwargs = dict((key, match.group(key)) for key in ('formid', 'hour'))
        kwargs['hour'] = int(kwargs['hour'])
        kwargs['run'] = run
        return cls(**kwargs)

    @property
    def id(self):
        return 'grib{0.run.hour:02d}{0.formid}{0.hour}'.format(self)

    @property
    def url(self):
        return self.run.url + 'gfs.t{0.run.hour:02d}z.pgrb2{0.formid}{0.hour}'.format(self)

class DatasetIndex(object):
    """
    An index of datasets available on the GFS servers. When constructed, the
    index is empty. Use :py:meth:`refresh` to build the index. If
    *refresh_immediately* is *True* then :py:meth:`refresh` is called on
    construction using *limit_runs* as its argument.

    .. py:attribute:: index

        A dictionary-like object keyed by :py:class:`RunRecord` instances. Use
        :py:attr:`latest_run` to get the latest run. Each value is a dictionary-like
        object whose keys correspond to the "formid" of the GRIB (i.e. ``f`` or
        ``bf``) and whose values are lists of :py:class:`GribRecord` instances
        sorted by ascending forcast hours.

        .. seealso:: http://www.nco.ncep.noaa.gov/pmb/products/gfs/

    .. py:attribute:: latest_run

        A :py:class:`RunRecord` instance specifying the latest run recorded in
        :py:attr:`index`. If no runs have yet been fetched, this is *None*.

    """
    def __init__(self, refresh_immediately=True, limit_runs=1):
        self.index = { }
        self.latest_run = None
        if refresh_immediately:
            self.refresh(limit_runs=limit_runs)

    def refresh(self, limit_runs=1):
        """Explicitly refresh the index of datasets and contained GRIB files.

        If *limit_runs* is not *None* then only the newest *limit_runs* will be
        considered. If *limit_runs* is larger than the number of runs on the
        server then all runs are considered.
        """
        # Fetch the index from the server
        LOG.info('Fetching run index from {0}'.format(DATA_ROOT))
        resp = requests.get(DATA_ROOT)

        # Form a set of all runs in the index.
        runs = set(
            RunRecord.from_match(run_match)
            for run_match in GFS_RUN_PATTERN.finditer(resp.text)
        )
        LOG.info('Discovered {0} runs(s)'.format(len(runs)))

        # Sort them newest first, oldest last. Note that this works because the
        # RunRecord is a subclass of tuple and so we end up sorting by year
        # then month then day and then hour. Isn't Python wonderful?
        runs = sorted(runs, reverse=True)

        # Do we want to only consider some runs?
        if limit_runs is not None:
            LOG.info('Using only the {0} newest run(s)'.format(limit_runs))
            runs = runs[:limit_runs]

        # For each run...
        self.index = { }
        self.latest_run = runs[0] if len(runs) > 0 else None
        for run in runs:
            # Fetch the index from the server
            LOG.info('Fetching run {0.id} index from {0.url}'.format(run))
            resp = requests.get(run.url)

            # Find all the GRIB files in the index
            gribs = set(
                GribRecord.from_match(run, grib_match)
                for grib_match in GRIB_PATTERN.finditer(resp.text)
            )
            LOG.info('Found {0} matching GRIB file(s)'.format(len(gribs)))

            # Split gribs into buckets based on their formid. Each bucket is
            # sorted by ascending hour.
            gribs_by_formid = dict(
                (k, sorted(v, key=lambda g: g.hour))
                for k, v in itertools.groupby(
                    sorted(gribs, key=lambda g: g.formid),
                    key=lambda g: g.formid
                )
            )

            # Update index
            self.index[run] = gribs_by_formid

    def latest_gribs_of_type(self, formid):
        """Return a sequence of :py:class:`GribRecord`s with the specified
        *formid* ordered by ascending forcast hour and which correspond to the
        latest run in the index.

        Raises *KeyError* if *formid* is invalid and *IndexError* if the index
        is empty.

        """
        if self.latest_run is None:
            raise IndexError('No latest run. Has refresh() been called?')
        return self.index[self.latest_run][formid]
