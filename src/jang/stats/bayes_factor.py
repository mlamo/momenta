import copy
import logging
import numpy as np
import ultranest

from jang.io import NuDetectorBase, Parameters, Transient
from jang.io.neutrinos import BackgroundPoisson
from jang.stats.model import ModelNested, ModelNested_BkgOnly


def build_minimal_experiment(detector: NuDetectorBase):
    detector0 = copy.deepcopy(detector)
    for s in detector0.samples:
        if not isinstance(s.background, BackgroundPoisson):
            return None
        s.background.Noff = 0
        s.nobserved = 0
        s.events = []
    return detector0


def run_bkg(detector: NuDetectorBase, parameters: Parameters):
    
    model_bkg = ModelNested_BkgOnly(detector, parameters)
    sampler = ultranest.ReactiveNestedSampler(model_bkg.param_names, model_bkg.loglike, model_bkg.prior)
    result = sampler.run(show_status=False, viz_callback=False)
    
    return result


def compute_correction_tobkg(detector: NuDetectorBase, src: Transient, parameters: Parameters, return_error: bool = False):
    
    detector0 = build_minimal_experiment(detector)
    if detector0 is None:
        logging.getLogger("jang").warning("Cannot correct Bayes factor as one of the samples has non-Poisson background.")
        if return_error:
            return 0, 0
        return 0
    
    model0_bkg = ModelNested_BkgOnly(detector0, parameters)
    sampler0_bkg = ultranest.ReactiveNestedSampler(model0_bkg.param_names, model0_bkg.loglike, model0_bkg.prior)
    result0_bkg = sampler0_bkg.run(show_status=False, viz_callback=False)
    
    model0 = ModelNested(detector0, src, parameters)
    sampler0 = ultranest.ReactiveNestedSampler(model0.param_names, model0.loglike, model0.prior)
    result0 = sampler0.run(show_status=False, viz_callback=False)

    if return_error:
        return result0["logz"] - result0_bkg["logz"], np.sqrt(result0["logzerr"]**2 + result0_bkg["logzerr"]**2)
    return result0["logz"] - result0_bkg["logz"]


def compute_log_bayes_factor_tobkg(result: dict, detector: NuDetectorBase, src: Transient, parameters: Parameters, corrected: bool = True, return_error: bool = False):
    
    result_bkg = run_bkg(detector, parameters)
    logB = result["logz"] - result_bkg["logz"]
    if corrected:
        logB_corr, err_logB_corr = compute_correction_tobkg(detector, src, parameters, return_error=True)
    else:
        logB_corr = err_logB_corr = 0
    logB -= logB_corr
        
    if return_error:
        err_logB = np.sqrt(result["logzerr"]**2 + result_bkg["logzerr"]**2 + err_logB_corr**2)
        return logB, err_logB
    return logB
