"""Defining the neutrino detectors and all related objects (samples, acceptances...)."""

import abc
import itertools
import logging
import os
import pymc as pm
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import List, Optional, Tuple, Union

import astropy
import astropy.coordinates
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import yaml
from astropy.units import Quantity, Unit, deg
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
from scipy.stats import gamma, multivariate_normal, truncnorm

import jang.stats.pdfs as pdf

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=scipy.integrate.IntegrationWarning)


def infer_uncertainties(input_array: Union[float, np.ndarray], nsamples: int, correlation: Optional[float] = None) -> np.ndarray:
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
        self.acceptances = {}

    def evaluate(self, energy: float | np.ndarray):
        return 0
    
    def compute_acceptance(self, fluxcomponent, *args):
        def func(x: float, *args):
            return fluxcomponent.evaluate(np.exp(x)) * self.evaluate(np.exp(x), *args) * np.exp(x)
        return quad(func, np.log(fluxcomponent.emin), np.log(fluxcomponent.emax), limit=500, args=args)[0]
    
    def to_acceptance(self, fluxcomponent, nside: int):
        return self.compute_acceptance(fluxcomponent) * np.ones(hp.nside2npix(nside))
    
    def get_acceptance(self, fluxcomponent, nside: int):
        if str(fluxcomponent) not in self.acceptances:
            self.acceptances[str(fluxcomponent)] = self.to_acceptance(fluxcomponent, nside)
        acc = hp.ud_grade(self.acceptances[str(fluxcomponent)], nside)
        return acc
        

class EffectiveAreaAllSky(EffectiveAreaBase):
    
    def __init__(self, csvfile: str):
        super().__init__()
        self.func = None
        self.read_csv(csvfile)
        
    def read_csv(self, csvfile: str):
        x, y = np.loadtxt(csvfile, delimiter=',').T
        self.func = interp1d(x, y, bounds_error=False, fill_value=0)
        
    def evaluate(self, energy: float | np.ndarray):
        return self.func(energy)
    
    def to_acceptance(self, flux, nside: int):
        acc = self.compute_acceptance(flux)
        return acc * np.ones(hp.nside2npix(nside))


class Background(abc.ABC):
    @abc.abstractmethod
    def prepare_toys(self, ntoys: int):
        pass

    @property
    @abc.abstractmethod
    def nominal(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass


class BackgroundFixed(Background):
    def __init__(self, b0: float):
        self.b0 = b0

    def prepare_toys(self, ntoys: int):
        return self.b0 * np.ones(ntoys)

    @property
    def nominal(self):
        return self.b0

    def __repr__(self):
        return f"{self.b0:.2e}"


class BackgroundGaussian(Background):
    def __init__(self, b0: float, error_b: float):
        self.b0, self.error_b = b0, error_b

    def prepare_toys(self, ntoys: int):
        return truncnorm.rvs(-self.b0 / self.error_b, np.inf, loc=self.b0, scale=self.error_b, size=ntoys)

    @property
    def nominal(self):
        return self.b0

    def __repr__(self):
        return f"{self.b0:.2e} +/- {self.error_b:.2e}"


class BackgroundPoisson(Background):
    def __init__(self, Noff: int, alpha_offon: int):
        self.Noff, self.alpha_offon = Noff, alpha_offon

    def prepare_toys(self, ntoys: int):
        return gamma.rvs(self.Noff + 1, scale=1 / self.alpha_offon, size=ntoys)

    @property
    def nominal(self):
        return self.Noff / self.alpha_offon

    def __repr__(self):
        return f"{self.nominal:.2e} = {self.Noff:d}/{self.alpha_offon:d}"


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


class NuSample:
    """Class to handle a given neutrino sample characterised by its name, observed events, expected background and PDFs."""

    def __init__(self, name: str = None, shortname: str = None):
        self.acceptances = {}
        self.effective_area = None
        self.name = name
        self.shortname = shortname if shortname is not None else name
        self.energy_range = (None, None)
        self.nobserved = np.nan
        self.background = None
        self.events = None
        self.pdfs = {
            "signal": {"ang": None, "ene": None, "time": None},
            "background": {"ang": None, "ene": None, "time": None},
        }

    def set_energy_range(self, emin: float, emax: float):
        self.energy_range = (emin, emax)

    def set_effective_area(self, aeff):
        self.effective_area = aeff

    def set_observations(self, nobserved: int, bkg: Background):
        self.nobserved = nobserved
        self.background = bkg

    def set_events(self, events: List[NuEvent]):
        self.events = events

    def set_pdfs(
        self,
        sig_ang: pdf.AngularSignal = None,
        sig_ene: pdf.EnergySignal = None,
        sig_time: pdf.TimeSignal = None,
        bkg_ang: pdf.AngularBackground = None,
        bkg_ene: pdf.EnergyBackground = None,
        bkg_time: pdf.TimeBackground = None,
    ):
        if (sig_ang is None) ^ (bkg_ang is None):
            raise RuntimeError("One of the angular PDFs is missing!")
        if (sig_ene is None) ^ (bkg_ene is None):
            raise RuntimeError("One of the energy PDFs is missing!")
        if (sig_time is None) ^ (bkg_time is None):
            raise RuntimeError("One of the time PDFs is missing!")
        self.pdfs["signal"]["ang"] = sig_ang
        self.pdfs["signal"]["ene"] = sig_ene
        self.pdfs["signal"]["time"] = sig_time
        self.pdfs["background"]["ang"] = bkg_ang
        self.pdfs["background"]["ene"] = bkg_ene
        self.pdfs["background"]["time"] = bkg_time

    @property
    def log_energy_range(self) -> Tuple[float, float]:
        return np.log(self.energy_range[0]), np.log(self.energy_range[1])

    @property
    def log10_energy_range(self) -> Tuple[float, float]:
        return np.log10(self.energy_range[0]), np.log10(self.energy_range[1])


class ToyNuDet:
    """Class to handle toys related to detector systematics."""

    def __init__(
        self,
        nobserved: np.ndarray,
        nbackground: np.ndarray,
        var_acceptance: np.ndarray,
        events: Optional[List[List[NuEvent]]] = None,
    ):
        self.nobserved = np.array(nobserved)
        self.nbackground = np.array(nbackground)
        self.var_acceptance = np.array(var_acceptance)
        self.events = events

    def __str__(self):
        return "ToyNuDet: n(observed)=%s, n(background)=%s, var(acceptance)=%s, events=%s" % (
            self.nobserved,
            self.nbackground,
            self.var_acceptance,
            self.events,
        )


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
   
    def get_acceptances(self, fluxcomponent, nside):
        return [s.effective_area.get_acceptance(fluxcomponent, nside) for s in self.samples]
    
    def get_nonempty_acceptance_pixels(self, flux, nside: int):
        accs = [self.get_acceptances(c, nside) for c in flux.components]
        accs = np.apply_over_axes(np.sum, np.array(accs), [0, 1])
        return np.argwhere(accs > 0)

    def prepare_toys(self, ntoys: int = 0) -> List[ToyNuDet]:

        toys = []
        nobserved = np.array([s.nobserved for s in self.samples])
        background = np.array([s.background for s in self.samples])
        events = [s.events for s in self.samples]

        if np.any(np.isnan(nobserved)):
            raise RuntimeError("[NuDetector] The number of observed events is not correctly filled.")

        # if no toys
        if ntoys == 0:
            nbackground = [bkg.nominal for bkg in background]
            toy = ToyNuDet(nobserved, nbackground, np.ones(self.nsamples), events)
            return [toy]

        # acceptance toys
        toys_acceptance = multivariate_normal.rvs(mean=np.ones(self.nsamples), cov=self.error_acceptance, size=ntoys)
        for i in range(ntoys):
            while np.any(toys_acceptance[i] < 0):
                toys_acceptance[i] = multivariate_normal.rvs(mean=np.ones(self.nsamples), cov=self.error_acceptance)

        # background toys
        toys_background = np.array([bkg.prepare_toys(ntoys) for bkg in background]).T

        for i in range(ntoys):
            toys.append(ToyNuDet(nobserved, toys_background[i], toys_acceptance[i], events))
        return toys


class NuDetector(NuDetectorBase):
    """Class to handle the neutrino detector information."""

    def __init__(self, infile: Optional[Union[dict, str]] = None):
        super().__init__()
        self.earth_location = None
        self.error_acceptance_corr = None
        if infile is not None:
            self.load(infile)

    def load(self, rinput: Union[dict, str]):
        """Load the detector configuration from either
        - JSON file (format defined in the examples folder
        - dictionary object (with same format as JSON).
        """
        log = logging.getLogger("jang")

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
        if "earth_location" in data:
            unit = Unit(data["earth_location"]["units"])
            self.earth_location = astropy.coordinates.EarthLocation(
                lat=data["earth_location"]["latitude"] * unit,
                lon=data["earth_location"]["longitude"] * unit,
            )
        for i in range(data["nsamples"]):
            smp = NuSample(
                name=data["samples"]["names"][i],
                shortname=data["samples"]["shortnames"][i] if "shortnames" in data["samples"] else None,
            )
            data["samples"]["energyrange"] = np.array(data["samples"]["energyrange"], dtype=float)
            if data["samples"]["energyrange"].shape == (data["nsamples"], 2):
                smp.set_energy_range(*data["samples"]["energyrange"][i])
            elif data["samples"]["energyrange"].shape == (2,):
                smp.set_energy_range(*data["samples"]["energyrange"])
            else:
                raise RuntimeError("[NuDetector] Unknown format for energy range.")
            self._samples.append(smp)
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
        self.error_acceptance = infer_uncertainties(
            self.error_acceptance, self.nsamples, correlation=self.error_acceptance_corr
        )


class SuperNuDetector(NuDetectorBase):
    """Class to handle several detectors simultaneously."""

    def __init__(self, name: Optional[str] = None):
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
        log = logging.getLogger("jang")
        if det.name in [d.name for d in self.detectors]:
            log.error(
                "[SuperDetector] Detector with same name %s is already loaded in the SuperDetector. Skipped.",
                det.name,
            )
            return
        log.info("[SuperDetector] Detector %s is added to the SuperDetector.", det.name)
        self.detectors.append(det)
        self.error_acceptance = block_diag(*[d.error_acceptance for d in self.detectors])
