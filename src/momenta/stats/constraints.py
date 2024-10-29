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

import copy
import numpy as np

from collections import defaultdict
from scipy.stats import gaussian_kde
from ultranest.integrator import resample_equal

from momenta.io import NuDetectorBase, Transient, Parameters
from momenta.stats.model import calculate_deterministics
from momenta.stats.run import run_ultranest
from momenta.utils.flux import FluxFixedPowerLaw


def upperlimit_from_sample(sample: np.ndarray, CL: float = 0.90) -> float:
    """Return upper limit at a given confidence level from a list of values

    Args:
        sample (np.ndarray): list of values
        CL (float, optional): desired confidence level. Defaults to 0.90.

    Returns:
        float: upper limit
    """

    x = np.array(sample).flatten()
    return np.percentile(x, 100 * CL)


def get_limits(samples: dict, model, CL: float = 0.90) -> dict[str, float]:
    """Compute all upper limits at a given confidence level, adding all relevant astro quantities.

    Args:
        samples (dict): dictionary of samples (output of sampling algorithm)
        model: model being used
        CL (float, optional): desired confidence level. Defaults to 0.90.

    Returns:
        dict[str, float]: dictionary of upper limits
    """

    samples.update(calculate_deterministics(samples, model))

    limits = {}
    for n, s in samples.items():
        limits[n] = upperlimit_from_sample(s, CL)
    return limits


def get_limits_with_uncertainties(weighted_samples: dict, model, CL: float = 0.90) -> dict[str, tuple[float]]:
    """Compute all upper limits at a given confidence level, adding all relevant astro quantities.

    Args:
        samples (dict): dictionary of weighted samples (output of sampling algorithm)
        model: model being used
        CL (float, optional): desired confidence level. Defaults to 0.90.

    Returns:
        dict[str, tuple[float]]: dictionary of upper limits with estimated error
    """

    limits = defaultdict(list)
    for weights in weighted_samples["bootstrapped_weights"].transpose():
        samples = {}
        for n, p in weighted_samples["points"].items():
            samples[n] = resample_equal(p, weights)
        _limits = get_limits(samples, model, CL)
        for n in _limits.keys():
            limits[n].append(_limits[n])

    res = {}
    for n in limits.keys():
        res[n] = (np.average(limits[n]), np.std(limits[n]))
    return res


def compute_differential_limits(detector: NuDetectorBase, src: Transient, parameters: Parameters, bins_energy: np.ndarray, spectral_index: float = 1):

    limits = []
    pars = copy.deepcopy(parameters)
    for ll, ul in zip(bins_energy[:-1], bins_energy[1:]):
        pars.flux = FluxFixedPowerLaw(ll, ul, spectral_index)
        model, result = run_ultranest(detector, src, pars)
        limits.append(get_limits(result["samples"], model)["flux0_norm"])
    return limits


def get_bestfit(sample: np.ndarray, xmin: float = 0, xmax: float = None):

    # getting PDF using KDE
    if xmin is None:
        xmin = np.min(sample)
    if xmax is None:
        xmax = np.max(sample)
    f = gaussian_kde(sample)
    x = np.linspace(xmin, xmax, 20000)
    y = f.evaluate(x)

    return x[np.argmax(y)]


def get_hpd_interval(sample: np.ndarray, CL: float = 0.90, xmin: float = 0, xmax: float = None):

    # getting PDF using KDE
    if xmin is None:
        xmin = np.min(sample)
    if xmax is None:
        xmax = np.max(sample)
    f = gaussian_kde(sample)
    x = np.linspace(xmin, xmax, 20000)
    y = f.evaluate(x)
    y /= np.sum(y)

    # getting all values in the HPD range
    isort = np.flipud(np.argsort(y))
    cumsum = 0
    idx_hpd = []
    for i, _y in zip(isort, y[isort]):
        cumsum += _y
        idx_hpd.append(i)
        if cumsum >= CL:
            break
    idx_hpd.sort()

    # getting HPD intervals
    modes = []
    ilow = idx_hpd[0]
    iprev = idx_hpd[0]
    for i in idx_hpd:
        if i > iprev + 1:
            modes.append([ilow, iprev])
            ilow = i
        iprev = i
    modes.append([ilow, idx_hpd[-1]])

    modes = x[np.array(modes)]
    return modes
