import logging
import numpy as np
import contextlib
import os
import sys

from jang.io import NuDetectorBase, GW, Parameters
import jang.stats.model_emcee as m_emcee
import jang.stats.model_multinest as m_multinest
import jang.stats.model_ultranest as m_ultranest
from jang.utils.conversions import solarmass_to_erg


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


def upperlimit_from_sample(sample: np.ndarray, CL: float = 0.90):
    x = np.array(sample).flatten()
    return np.percentile(x, 100*CL)
   
   
def calculate_deterministics(samples, model):
    det = {}
    if "distance_scaling" not in model.toys_src.columns:
        return det
    itoys = samples["itoy"]
    nsamples = len(itoys)
    distance_scaling = model.toys_src.iloc[itoys]["distance_scaling"].to_numpy()
    norms = np.array([samples[f"flux{i}_norm"] for i in range(model.flux.ncomponents)])
    if model.flux.nshapes > 0:
        shapes = np.array([samples[f"flux{i}_{s}"] for i, c in enumerate(model.flux.components) for s in c.shape_names])
        det["eiso"] = np.empty(nsamples)
        for isample in range(nsamples):
            model.flux.set_shapes(shapes[:,isample])
            det["eiso"][isample] = np.sum(norms * model.flux.flux_to_eiso(distance_scaling[isample]))        
    else:
        det["eiso"] = np.sum(norms + model.flux.flux_to_eiso(distance_scaling), axis=0)
    if "theta_jn" not in model.toys_src.columns:
        return det
    theta_jn = model.toys_src.iloc[itoys]["theta_jn"].to_numpy()
    det["etot"] = det["eiso"] / model.parameters.jet.etot_to_eiso(theta_jn)
    if "radiated_energy" not in model.toys_src.columns:
        return det
    radiated_energy = model.toys_src.iloc[itoys]["radiated_energy"].to_numpy()
    det["fnu"] = det["etot"] / (radiated_energy * solarmass_to_erg)
    return det
    
    
def get_limits(detector: NuDetectorBase, gw: GW, parameters: Parameters, method="pymc") -> tuple[float]:
    
    if method == "emcee":
        model = m_emcee.prepare_model(detector, gw, parameters)
        samples = m_emcee.run_mcmc(model)
    elif method == "multinest":
        with redirect_stdout(os.devnull):  # MultiNest can be quite noisy, we silent it!
            model = m_multinest.prepare_model(detector, gw, parameters)
            samples = m_multinest.run_mcmc(model)
    elif method == "ultranest":
        model = m_multinest.prepare_model(detector, gw, parameters)
        samples = m_ultranest.run_mcmc(model)
    samples.update(calculate_deterministics(samples, model))
    
    limit_flux = []
    for n in model.param_names:
        if n.startswith("flux"):
            limit_flux.append(upperlimit_from_sample(samples[n], 0.90))
        logging.getLogger("jang").info("[Limits(MCMC-%s)] %s, %s, %s, limit(%s) = %.3e", method, gw.name, detector.name, parameters.flux, n, limit_flux[-1])
    limit_eiso = upperlimit_from_sample(samples["eiso"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC-%s)] %s, %s, %s, limit(Eiso) = %.3e", method, gw.name, detector.name, parameters.flux, limit_eiso)
    limit_etot = upperlimit_from_sample(samples["etot"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC-%s)] %s, %s, %s, limit(Etot) = %.3e", method, gw.name, detector.name, parameters.flux, limit_etot)
    limit_fnu = upperlimit_from_sample(samples["fnu"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC-%s)] %s, %s, %s, limit(fnu) = %.3e", method, gw.name, detector.name, parameters.flux, limit_fnu)
    
    import matplotlib.pyplot as plt
    plt.hist2d(samples["flux0_gamma"], samples["flux0_norm"])
    plt.xlabel("Spectral index")
    plt.ylabel("Flux normalisation")
    plt.savefig("test.png")
    
    return limit_flux, limit_eiso, limit_etot, limit_fnu
