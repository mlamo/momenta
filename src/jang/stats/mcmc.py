import logging
import numpy as np
import contextlib

from jang.io import NuDetectorBase, GW, Parameters
import jang.stats.model_emcee as m_emcee
import jang.stats.model_multinest as m_multinest
import jang.stats.model_ultranest as m_ultranest
from jang.utils.conversions import solarmass_to_erg


def upperlimit_from_sample(sample: np.ndarray, CL: float = 0.90):
    x = np.array(sample).flatten()
    return np.percentile(x, 100*CL)
   
   
def calculate_deterministics(samples, model):
    det = {}
    if "distance_scaling" not in model.toys_src.columns:
        return det
    itoys = samples["itoy"]
    distance_scaling = model.toys_src.iloc[itoys]["distance_scaling"].to_numpy()
    norms = np.array([samples[f"phi{i}"] for i in range(model.flux.ncomponents)])
    det["eiso"] = np.sum(norms * model.flux.flux_to_eiso(distance_scaling), axis=0)
    if "theta_jn" not in model.toys_src.columns:
        return det
    theta_jn = model.toys_src.iloc[itoys]["theta_jn"].to_numpy()
    det["etot"] = det["eiso"] / model.jetmodel.etot_to_eiso(theta_jn)
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
        with contextlib.redirect_stdout(None):  # MultiNest can be quite noisy, we silent it!
            model = m_multinest.prepare_model(detector, gw, parameters)
            samples = m_multinest.run_mcmc(model)
    elif method == "ultranest":
        model = m_multinest.prepare_model(detector, gw, parameters)
        samples = m_ultranest.run_mcmc(model)
    samples.update(calculate_deterministics(samples, model))
    
    limit_phi = []
    for i in range(parameters.flux.ncomponents):
        limit_phi.append(upperlimit_from_sample(samples[f"phi{i}"], 0.90))
        logging.getLogger("jang").info("[Limits(MCMC-%s)] %s, %s, %s, limit(Flux%d) = %.3e", method, gw.name, detector.name, parameters.flux, i, limit_phi[-1])
    limit_eiso = upperlimit_from_sample(samples["eiso"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(Eiso) = %.3e", gw.name, detector.name, parameters.flux, limit_eiso)
    limit_etot = upperlimit_from_sample(samples["etot"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(Etot) = %.3e", gw.name, detector.name, parameters.flux, limit_etot)
    limit_fnu = upperlimit_from_sample(samples["fnu"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(fnu) = %.3e", gw.name, detector.name, parameters.flux, limit_fnu)
    
    return limit_phi, limit_eiso, limit_etot, limit_fnu
