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
import itertools
import logging
import os
import warnings

import healpy as hp
import numpy as np
import scipy.integrate
import yaml
from scipy.integrate import quad, trapezoid
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.linalg import block_diag
from scipy.stats import gamma, truncnorm

import astropy.coordinates
import astropy.time
from astropy.units import deg, rad

import momenta.stats.pdfs as pdf


warnings.filterwarnings("ignore", category=scipy.integrate.IntegrationWarning)


def infer_uncertainties(input_array: float | np.ndarray, nsamples: int, correlation: float | None = None) -> np.ndarray:
    """Infer uncertainties based on an input array that could be:
    - 0-D (same error for each sample)
    - 1-D (one error per sample)
    - 2-D (correlation matrix)
    """
    if input_array is None:
        return None
    input_array = np.array(input_array)
    correlation_matrix = (correlation if correlation is not None else 0) * np.ones((nsamples, nsamples))
    np.fill_diagonal(correlation_matrix, 1)
    # if uncertainty is a scalar (error for all samples)
    if input_array.ndim == 0:
        return input_array * correlation_matrix * input_array
    # if uncertainty is a vector (error for each sample)
    if input_array.shape == (nsamples,):
        return np.array([[input_array[i] * correlation_matrix[i, j] * input_array[j] for i in range(nsamples)] for j in range(nsamples)])
    # if uncertainty is a covariance matrix
    if input_array.shape == (nsamples, nsamples):
        return input_array
    raise RuntimeError("The size of uncertainty_acceptance does not match with the number of samples")


class EffectiveAreaBase:
    """Class to handle detector effective area for a given sample and neutrino flavour.
    This default class handles only energy-dependent effective area."""

    def __init__(self):
        self._acceptances = {}

    def evaluate(self, energy: float | np.ndarray, ipix: int, nside: int):
        """Evaluate the effective area for a given energy, pixel index, and skymap resolution."""
        return 0

    def _compute_acceptance(self, fluxcomponent, ipix: int, nside: int):
        """Compute the acceptance integrating the effective area x flux in log scale between emin and emax.
        Only used internally by `compute_acceptance_map`, may be overriden in a inheriting class."""
        def func(x: float):
            return fluxcomponent.evaluate(np.exp(x)) * self.evaluate(np.exp(x), ipix, nside) * np.exp(x)

        return quad(func, np.log(fluxcomponent.emin), np.log(fluxcomponent.emax), limit=500)[0]

    def compute_acceptance_map(self, fluxcomponent, nside: int):
        """Compute the acceptance map for a given flux component, iterating over all pixels."""
        return np.array([self._compute_acceptance(fluxcomponent, ipix, nside) for ipix in range(hp.nside2npix(nside))])

    def compute_acceptance_maps(self, fluxcomponents, nside: int):
        """Compute the acceptance maps for a list of flux components, iterating over all components and pixels.
        May be overriden by a smarter implementation for specific cases where computation can be optimized."""
        return [self.compute_acceptance_map(c, nside) for c in fluxcomponents]

    def get_acceptance_map(self, fluxcomponent, nside: int):
        """Get the acceptance for a given flux component. 
        If the component `store` attribute is 'exact', it is retrieved from the `acceptances` dictionary (added there if not yet available).
        If the component `store` attribute is 'interpolate', the dictionary with the interpolation function (+inputs) is returned.
        """
        if fluxcomponent.store == "exact":
            if (str(fluxcomponent), nside) not in self._acceptances:
                self._acceptances[(str(fluxcomponent), nside)] = self.compute_acceptance_map(fluxcomponent, nside)
            return self._acceptances[(str(fluxcomponent), nside)]
        if fluxcomponent.store == "interpolate":
            if (str(fluxcomponent), nside) not in self._acceptances:
                accs = {str(c): a for c, a in zip(fluxcomponent.grid.flatten(), self.compute_acceptance_maps(fluxcomponent.grid.flatten(), nside))}
                def f(_c):
                    return accs[str(_c)]
                accs = np.vectorize(f, signature="()->(n)")(fluxcomponent.grid)
                grid = [*fluxcomponent.shapevar_grid, np.arange(hp.nside2npix(nside))]
                self._acceptances[(str(fluxcomponent), nside)] = RegularGridInterpolator(grid, accs)
            return self._acceptances[(str(fluxcomponent), nside)]
        return self.compute_acceptance_map(fluxcomponent, nside)

    def get_acceptance(self, fluxcomponent, ipix: int, nside: int):
        """Get the acceptance"""
        if fluxcomponent.store == "exact":
            return self.get_acceptance_map(fluxcomponent, nside)[ipix]
        if fluxcomponent.store == "interpolate":
            return self.get_acceptance_map(fluxcomponent, nside)([*fluxcomponent.shapevar_values, ipix])
        return self._compute_acceptance(fluxcomponent, ipix, nside)


class EffectiveAreaAllSky(EffectiveAreaBase):

    def __init__(self):
        super().__init__()
        self.func = None

    def read_csv(self, csvfile: str):
        x, y = np.loadtxt(csvfile, delimiter=",").T
        self.func = interp1d(x, y, bounds_error=False, fill_value=0)

    def evaluate(self, energy: float | np.ndarray, ipix: int, nside: int):
        return self.func(energy)

    def compute_acceptance_map(self, fluxcomponent, nside):
        acc = self._compute_acceptance(fluxcomponent, 0, nside) * np.ones(hp.nside2npix(nside))
        return acc


class EffectiveAreaDeclinationDep(EffectiveAreaBase):

    def __init__(self):
        super().__init__()
        self.mapping = {}

    def evaluate(self, energy: float | np.ndarray, ipix: int, nside: int):
        if nside not in self.mapping:
            self.mapping[nside] = self.map_ipix_to_declination(nside)
        return self.func(energy, self.mapping[nside][ipix])

    def compute_acceptance_map(self, fluxcomponent, nside):
        if nside not in self.mapping:
            self.mapping[nside] = self.map_ipix_to_declination(nside)
        acc = np.zeros(hp.nside2npix(nside))
        for dec, ipix in zip(*np.unique(self.mapping[nside], return_index=True)):
            acc[self.mapping[nside] == dec] = self._compute_acceptance(fluxcomponent, ipix, nside)
        return acc

    def map_ipix_to_declination(self, nside):
        ipix = np.arange(hp.nside2npix(nside))
        _, dec = hp.pix2ang(nside, ipix, lonlat=True)
        return dec


class EffectiveAreaAltitudeDep(EffectiveAreaBase):

    def __init__(self):
        super().__init__()
        self.func = None
        self.mapping = {}

    def read(self):
        bins_logenergy = ...  # shape (M,)
        bins_altitude = ...  # shape (N,)
        aeff = ...  # shape (M,N)
        self.func = RegularGridInterpolator((bins_logenergy, bins_altitude), aeff, bounds_error=False, fill_value=0)

    def set_location(self, time, lat_deg, lon_deg):
        self.obstime = time
        self.location = astropy.coordinates.EarthLocation(lat=lat_deg * deg, lon=lon_deg * deg)
        self.mapping = {}

    def evaluate(self, energy: float | np.ndarray, ipix: int, nside: int):
        if nside not in self.mapping:
            self.mapping[nside] = self.map_ipix_to_altitude(nside)
        return self.func((np.log10(energy), self.mapping[nside][ipix]))

    def compute_acceptance_map(self, fluxcomponent, nside):
        if nside not in self.mapping:
            self.mapping[nside] = self.map_ipix_to_altitude(nside)
        acc = np.zeros(hp.nside2npix(nside))
        for alt, ipix in zip(*np.unique(self.mapping[nside], return_index=True)):
            acc[self.mapping[nside] == alt] = self._compute_acceptance(fluxcomponent, ipix, nside)
        return acc

    def map_ipix_to_altitude(self, nside):
        ipix = np.arange(hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, ipix, lonlat=True)
        coords_eq = astropy.coordinates.SkyCoord(ra=ra * deg, dec=dec * deg, frame="icrs")
        coords_loc = coords_eq.transform_to(astropy.coordinates.AltAz(obstime=self.obstime, location=self.location))
        return coords_loc.alt.deg


class Background(abc.ABC):
    @property
    @abc.abstractmethod
    def nominal(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    def prior_transform(self):
        """Function used by multinest"""
        pass


class BackgroundFixed(Background):
    def __init__(self, b0: float):
        self.b0 = b0

    @property
    def nominal(self):
        return self.b0

    def __repr__(self):
        return f"{self.b0:.2e}"

    def prior_transform(self, x):
        return self.b0


class BackgroundGaussian(Background):
    def __init__(self, b0: float, error_b: float):
        self.b0, self.error_b = b0, error_b
        self.func = truncnorm(-self.b0 / self.error_b, np.inf, loc=self.b0, scale=self.error_b)

    @property
    def nominal(self):
        return self.b0

    def __repr__(self):
        return f"{self.b0:.2e} +/- {self.error_b:.2e}"

    def prior_transform(self, x):
        return self.func.ppf(x)


class BackgroundPoisson(Background):
    def __init__(self, Noff: int, alpha_offon: int):
        self.Noff, self.alpha_offon = Noff, alpha_offon
        self.func = gamma(self.Noff + 1, scale=1 / self.alpha_offon)

    @property
    def nominal(self):
        return self.Noff / self.alpha_offon

    def __repr__(self):
        return f"{self.nominal:.2e} = {self.Noff:.0f}/{self.alpha_offon:.2e}"

    def prior_transform(self, x):
        return self.func.ppf(x)


class NuEvent:
    """Class to handle a single neutrino candidate."""

    def __init__(
        self,
        dt: float = np.nan,
        ra: float = np.nan,
        dec: float = np.nan,
        energy: float = np.nan,
        sigma: float = np.nan,
        altitude: float = np.nan,
        azimuth: float = np.nan,
    ):
        """Event is defined by:
        - dt = t(neutrino)-t(GW) [in seconds]
        - ra/dec = reconstructed equatorial directions [in radians]
        - energy = reconstructed energy [in GeV]
        - sigma = uncertainty on reconstructed direction [in radians]
        - altitude/azimuth = reconstructed local directions [in radians]
        """
        self.dt = dt
        self.ra = ra
        self.dec = dec
        self.energy = energy
        self.sigma = sigma
        self.altitude = altitude
        self.azimuth = azimuth

    def __repr__(self):
        r = f"Event(deltaT={self.dt:.0f} s, ra/dec={np.rad2deg(self.ra):.1f}/{np.rad2deg(self.dec):.1f} deg, energy={self.energy:.2g} GeV, "
        r += f"sigma={np.rad2deg(self.sigma):.2f} deg, alt/azi={np.rad2deg(self.altitude):.1f}/{np.rad2deg(self.azimuth):.1f} deg)"
        return r

    @property
    def sindec(self):
        return np.sin(self.dec)

    @property
    def sinalt(self):
        return np.sin(self.altitude)

    @property
    def log10energy(self):
        return np.log10(self.energy)

    @property
    def coords(self):
        return astropy.coordinates.SkyCoord(ra=self.ra * rad, dec=self.dec * rad)


class NuSample:
    """Class to handle a given neutrino sample characterised by its name, observed events, expected background and PDFs."""

    def __init__(self, name: str | None = None):
        self.name = name
        self.effective_area = None
        self.nobserved = np.nan
        self.background = None
        self._events = None
        self._pdfs = {
            "signal": {"ang": None, "ene": None, "time": None},
            "background": {"ang": None, "ene": None, "time": None},
        }

    def set_effective_area(self, aeff):
        self.effective_area = aeff

    def set_observations(self, nobserved: int, bkg: Background):
        self.nobserved = nobserved
        self.background = bkg

    def set_events(self, events: list[NuEvent]):
        assert len(events) == self.nobserved or events is None
        self._events = events
        
    @property
    def events(self):
        return self._events

    def set_pdfs(
        self,
        sig_ang: pdf.AngularSignal | None = None,
        sig_ene: pdf.EnergySignal | None = None,
        sig_time: pdf.TimeSignal | None = None,
        bkg_ang: pdf.AngularBackground | None = None,
        bkg_ene: pdf.EnergyBackground | None = None,
        bkg_time: pdf.TimeBackground | None = None,
    ):
        if (sig_ang is None) ^ (bkg_ang is None):
            raise RuntimeError("One of the angular PDFs is missing!")
        if (sig_ene is None) ^ (bkg_ene is None):
            raise RuntimeError("One of the energy PDFs is missing!")
        if (sig_time is None) ^ (bkg_time is None):
            raise RuntimeError("One of the time PDFs is missing!")
        self._pdfs["signal"]["ang"] = sig_ang
        self._pdfs["signal"]["ene"] = sig_ene
        self._pdfs["signal"]["time"] = sig_time
        self._pdfs["background"]["ang"] = bkg_ang
        self._pdfs["background"]["ene"] = bkg_ene
        self._pdfs["background"]["time"] = bkg_time

    def compute_background_probability(self, ev):
        pbkg = 1
        if self._pdfs["background"]["ang"] is not None:
            pbkg *= self._pdfs["background"]["ang"](ev)
        if self._pdfs["background"]["ene"] is not None:
            pbkg *= self._pdfs["background"]["ene"](ev)
        return pbkg
    
    def compute_signal_probability(self, ev, fluxcomponent, ra_src, dec_src):
        psig = 1
        if self._pdfs["signal"]["ang"] is not None:
            psig *= self._pdfs["signal"]["ang"](ev, ra_src, dec_src)
        if self._pdfs["signal"]["ene"] is not None:
            psig *= self._pdfs["signal"]["ene"](ev, fluxcomponent)
        return psig


class NuDetectorBase(abc.ABC):
    """Class to handle the neutrino detector information."""

    def __init__(self):
        self.name = None
        self._samples = []
        self.error_acceptance = None

    @property
    def samples(self):
        return self._samples

    @property
    def nsamples(self):
        return len(self._samples)

    def get_acceptance_maps(self, fluxcomponent, nside):
        return [s.effective_area.get_acceptance_map(fluxcomponent, nside) for s in self.samples]


class NuDetector(NuDetectorBase):
    """Class to handle the neutrino detector information."""

    def __init__(self, infile: dict | str | None = None):
        super().__init__()
        self.earth_location = None
        self.error_acceptance_corr = None
        if infile is not None:
            self.load(infile)

    def load(self, rinput: dict | str):
        """Load the detector configuration from either
        - JSON file (format defined in the examples folder
        - dictionary object (with same format as JSON).
        """
        log = logging.getLogger("momenta")

        if isinstance(rinput, str):  # pragma: no cover
            assert os.path.isfile(rinput)
            with open(rinput) as f:
                data = yaml.safe_load(f)
            log.info("[NuDetector] Object is loaded from the file %s.", rinput)
        elif isinstance(rinput, dict):
            data = rinput
            log.info("[NuDetector] Object is loaded from a dictionary object.")
        else:
            raise TypeError("Unknown input format for Detector constructor")
        self.name = data["name"]
        for s in data["samples"]:
            self._samples.append(NuSample(name=s))
        if "errors" in data:
            self.error_acceptance = data["errors"]["acceptance"]
            self.error_acceptance_corr = data["errors"]["acceptance_corr"]
        else:
            self.error_acceptance = 0
            self.error_acceptance_corr = 0
        self.check_errors_validity()

    def set_observations(self, nobserved: list, background: list):
        if len(nobserved) != self.nsamples:
            raise RuntimeError("[NuDetector] Incorrect size for nobserved as compared to the number of samples.")
        if len(background) != self.nsamples:
            raise RuntimeError("[NuDetector] Incorrect size for nbackground as compared to the number of samples.")
        for i, smp in enumerate(self.samples):
            smp.set_observations(nobserved[i], background[i])

    def set_effective_areas(self, aeffs: list[EffectiveAreaBase]):
        for i, smp in enumerate(self.samples):
            smp.set_effective_area(aeffs[i])

    def check_errors_validity(self):
        self.error_acceptance = infer_uncertainties(self.error_acceptance, self.nsamples, correlation=self.error_acceptance_corr)


class SuperNuDetector(NuDetectorBase):
    """Class to handle several detectors simultaneously."""

    def __init__(self, name: str | None = None):
        super().__init__()
        self.name = name
        self.detectors = []

    @property
    def samples(self):
        return itertools.chain.from_iterable([det.samples for det in self.detectors])

    @property
    def nsamples(self):
        return sum([det.nsamples for det in self.detectors])

    def add_detector(self, det: NuDetector):
        log = logging.getLogger("momenta")
        if det.name in [d.name for d in self.detectors]:
            log.error(
                "[SuperDetector] Detector with same name %s is already loaded in the SuperDetector. Skipped.",
                det.name,
            )
            return
        log.info("[SuperDetector] Detector %s is added to the SuperDetector.", det.name)
        self.detectors.append(det)
        self.error_acceptance = block_diag(*[d.error_acceptance for d in self.detectors])
