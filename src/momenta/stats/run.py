import numpy as np
import contextlib
import logging
import os

from momenta.io import NuDetectorBase, Transient, Parameters
from momenta.stats.model import ModelNested

from pymultinest.solve import solve as multinest_solve
import ultranest


logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)


@contextlib.contextmanager
def redirect_stdout(dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr e.g.:
    with sys.stdout_redirected(sys.stderr, os.devnull):
        ...
    """
    try:
        old = os.dup(1), os.dup(2)
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), 1)
        os.dup2(dest_file.fileno(), 1)
        yield
    finally:
        os.dup2(old[0], 1)
        os.dup2(old[1], 2)
        dest_file.close()
    
    
def run_emcee(detector: NuDetectorBase, src: Transient, parameters: Parameters):
    
    model = ModelEmcee(detector, src, parameters)

    nwalkers = 32
    sampler = emcee.EnsembleSampler(nwalkers, model.ndims, model.log_prob)
    state = sampler.run_mcmc(model.get_starting_points(nwalkers), 5000)
    sampler.reset()
    sampler.run_mcmc(state, 20000)
    
    result = {}
    result["samples"] = {k: v for k, v in zip(model.param_names, sampler.get_chain(flat=True).transpose())}    
    return model, result


def run_multinest(detector: NuDetectorBase, src: Transient, parameters: Parameters):
    
    model = ModelNested(detector, src, parameters)
    
    with redirect_stdout(os.devnull):
        result = multinest_solve(LogLikelihood=model.loglike, Prior=model.prior, n_dims=model.ndims, verbose=False, sampling_efficiency=0.1)
    
    result["samples"] = {k: v for k, v in zip(model.param_names, result["samples"].transpose()) if k.startswith("flux") or k=="itoy"}
    return model, result


def run_ultranest(detector: NuDetectorBase, src: Transient, parameters: Parameters):
    
    model = ModelNested(detector, src, parameters)
    
    sampler = ultranest.ReactiveNestedSampler(model.param_names, model.loglike, model.prior)
    result = sampler.run(show_status=False, viz_callback=False)
    
    result["samples"] = {k: v for k, v in zip(model.param_names, result["samples"].transpose()) if k.startswith("flux") or k=="itoy"}
    result["weighted_samples"] = {
        'weights': result["weighted_samples"]['bootstrapped_weights'].transpose(),
        'points' : {k: v for k, v in zip(model.param_names, result["weighted_samples"]["points"].transpose()) if k.startswith("flux") or k=="itoy"}
    }
    return model, result
