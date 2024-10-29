"""
    Copyright (C) 2024  Mathieu Lamoureux

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import abc
import astropy.coordinates
import astropy.time
import astropy.units as u
import healpy as hp
import logging
import numpy as np
import pandas as pd

from scipy.stats import vonmises

import momenta.utils.conversions


class Transient:

    def __init__(self, name: str = None, utc: astropy.time.Time | None = None, logger: str = "momenta"):
        self.name = name
        self.utc = utc
        self.logger = logger

    @property
    def log(self):
        return logging.getLogger(self.logger)

    @abc.abstractmethod
    def prepare_prior_samples(self) -> pd.DataFrame:
        return


class PointSource(Transient):

    def __init__(
        self, ra_deg: float, dec_deg: float, err_deg: float, name: str = None, utc: astropy.time.Time | None = None, logger: str = "momenta"
    ):
        super().__init__(name, utc, logger)
        self.coords = astropy.coordinates.SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        self.err = err_deg * u.deg
        self.distance = None
        self.redshift = None

    def set_distance(self, distance):
        self.distance = distance
        self.redshift = momenta.utils.conversions.lumidistance_to_redshift(distance)

    def set_redshift(self, redshift):
        self.distance = momenta.utils.conversions.redshift_to_lumidistance(redshift)
        self.redshift = redshift

    def prepare_prior_samples(self, nside: int) -> pd.DataFrame:
        toys = {}
        if self.err == 0 * u.deg:
            toys["ra"] = [self.coords.ra.deg]
            toys["dec"] = [self.coords.dec.deg]
            if self.distance:
                toys["distance_scaling"] = [momenta.utils.conversions.distance_scaling(self.distance, self.redshift)]
        else:
            kappa = 1 / (self.err.to(u.rad).value) ** 2
            theta = vonmises.rvs(kappa, size=10000)
            phi = np.random.uniform(0, 2 * np.pi, size=10000)
            dra = np.arcsin(np.sin(theta) * np.cos(phi))
            ddec = np.arcsin(np.sin(theta) * np.sin(phi))
            toys["ra"] = self.coords.ra.deg + np.rad2deg(dra)
            toys["dec"] = self.coords.dec.deg + np.rad2deg(ddec)
        toys["ipix"] = hp.ang2pix(nside, toys["ra"], toys["dec"], lonlat=True)
        return pd.DataFrame(data=toys).to_records(index=False)
