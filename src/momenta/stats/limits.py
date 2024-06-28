import copy
import numpy as np

from collections import defaultdict
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
    return np.percentile(x, 100*CL)


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
    for weights in weighted_samples['weights']:
        samples = {}
        for n, p in weighted_samples['points'].items():
            samples[n] = resample_equal(p, weights)
        l = get_limits(samples, model, CL)
        for n in l.keys():
            limits[n].append(l[n])
    
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