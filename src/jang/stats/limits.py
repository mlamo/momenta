import numpy as np

from collections import defaultdict
from ultranest.integrator import resample_equal

from jang.stats.model import calculate_deterministics


def upperlimit_from_sample(sample: np.ndarray, CL: float = 0.90):
    x = np.array(sample).flatten()
    return np.percentile(x, 100*CL)


def get_limits(samples, model) -> dict[str, float]:
    
    samples.update(calculate_deterministics(samples, model))
    
    limits = {}
    for n, s in samples.items():
        limits[n] = upperlimit_from_sample(s, 0.90)    
    return limits


def get_limits_with_uncertainties(weighted_samples, model):
    
    limits = defaultdict(list)
    for weights in weighted_samples['weights']:
        samples = {}
        for n, p in weighted_samples['points'].items():
            samples[n] = resample_equal(p, weights)
        l = get_limits(samples, model)
        for n in l.keys():
            limits[n].append(l[n])
    
    res = {}
    for n in limits.keys():
        res[n] = (np.average(limits[n]), np.std(limits[n]))
    return res