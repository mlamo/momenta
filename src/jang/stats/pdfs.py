import abc
from typing import Callable

import numpy as np
from scipy.stats import norm


def angular_distance(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray, degrees1=False, degrees2=False):
    if degrees1:
        r1, d1 = np.deg2rad(ra1), np.deg2rad(dec1)
    else:
        r1, d1 = ra1, dec1
    if degrees2:
        r2, d2 = np.deg2rad(ra2), np.deg2rad(dec2)
    else:
        r2, d2 = ra2, dec2
    s = np.cos(r1 - r2) * np.cos(d1) * np.cos(d2) + np.sin(d1) * np.sin(d2)
    return np.arccos(np.clip(s, -1, 1))


class PDF:

    @abc.abstractmethod
    def __call__(self, evt):
        pass


class EnergySignal(PDF):
    """The standard energy signal PDF is a function f(ra,dec,E,flux)."""

    def __init__(self, func: Callable = None):
        self.func = func

    def __call__(self, evt, flux):
        return self.func(evt.ra, evt.dec, evt.energy, flux)


class AngularSignal(PDF):
    """The standard angular signal PDF is a function f(ra,dec,ra[src],dec[src],E)."""

    def __init__(self, func: Callable = None):
        self.func = func

    def __call__(self, evt, ra_src: float, dec_src: float, degrees_evt: bool = False):
        dpsi = angular_distance(evt.ra, evt.dec, ra_src, dec_src, degrees1=degrees_evt, degrees2=True)
        return self.func(dpsi, evt.energy)


class VonMisesSignal(AngularSignal):
    """A common angular signal PDF is Von Mises distribution f = VM(dpsi, sigma)."""

    def __init__(self):
        pass

    def __call__(self, evt, ra_src: float, dec_src: float, degrees_evt: bool = False):
        dpsi = angular_distance(evt.ra, evt.dec, ra_src, dec_src, degrees1=degrees_evt, degrees2=True)
        if evt.sigma > np.radians(7):
            kappa = 1.0 / evt.sigma**2
            return kappa * np.exp(kappa * np.cos(dpsi)) / (4 * np.pi * np.sinh(kappa))
        else:
            return 0.5 / np.pi / evt.sigma**2 * np.exp(-0.5 * (dpsi / evt.sigma) ** 2)


class EnergyBackground(PDF):
    """The standard energy background PDF is a function f(ra,dec,E)."""

    def __init__(self, func: Callable = None):
        self.func = func

    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)


class AngularBackground(PDF):
    """The standard angular background PDF is a function f(ra,dec,E)."""

    def __init__(self, func: Callable = None):
        self.func = func

    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)


class TimeSignal(PDF):
    """The standard time signal PDF is a function f(deltaT)."""

    def __init__(self, func: Callable = None):
        self.func = func

    def __call__(self, evt):
        return self.func(evt.dt)


class TimeBoxSignal(PDF):
    """A common time signal PDF is 1/dt for t0 <= t < t0+dt and 0 otherwise."""

    def __init__(self, t0: float = None, sigma_t: float = None):
        self.t0 = t0
        self.sigma_t = sigma_t

    def __call__(self, evt, t0: float = None, sigma_t: float = None):
        t0 = self.t0 if t0 is None else t0
        sigma_t = self.sigma_t if sigma_t is None else sigma_t
        return 1 / sigma_t * ((evt.dt >= t0) & (evt.dt < t0 + sigma_t))


class TimeGausSignal(PDF):
    """A commin time signal PDF is a normal distribution centered on t0."""

    def __init__(self, t0: float = None, sigma_t: float = None):
        self.t0 = t0
        self.sigma_t = sigma_t

    def __call__(self, evt, t0: float = None, sigma_t: float = None):
        t0 = self.t0 if t0 is None else t0
        sigma_t = self.sigma_t if sigma_t is None else sigma_t
        return norm.pdf(evt.dt, loc=t0, scale=sigma_t)


class TimeBackground(PDF):
    """The standard time background PDF is an uniform function f(deltaT) = 1/timewindow."""

    def __init__(self, timewindow_length: float):
        self.timewindow_length = timewindow_length

    def __call__(self, evt):
        return 1 / self.timewindow_length
