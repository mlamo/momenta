import numpy as np
import ultranest

from jang.io import NuDetectorBase, Parameters
from jang.stats.model import ModelNested_BkgOnly



def run_bkg(detector: NuDetectorBase, parameters: Parameters):
    
    model_bkg = ModelNested_BkgOnly(detector, parameters)
    sampler = ultranest.ReactiveNestedSampler(model_bkg.param_names, model_bkg.loglike, model_bkg.prior)
    result = sampler.run(show_status=False, viz_callback=False)
    
    return result


def compute_log_bayes_factor(result: dict, detector: NuDetectorBase, parameters: Parameters, return_error: bool = False):
    
    result_bkg = run_bkg(detector, parameters)
    logB = result["logz"] - result_bkg["logz"]
    if return_error:
        err_logB = np.sqrt(result["logzerr"]**2 + result_bkg["logzerr"]**2)
        return logB, err_logB
    return logB
