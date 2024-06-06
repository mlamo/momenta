import logging
import numpy as np

from jang.io import NuDetectorBase, GW, Parameters
import jang.mcmc.model_pymc as m_pymc
import jang.mcmc.model_emcee as m_emcee
import jang.mcmc.model_multinest as m_multinest


def upperlimit_from_sample(sample: np.ndarray, CL: float = 0.90):
    x = np.array(sample).flatten()
    return np.percentile(x, 100*CL)
   
    
def get_limits(detector: NuDetectorBase, gw: GW, parameters: Parameters, method="pymc") -> tuple[float]:
    
    if method == "pymc":
        model = m_pymc.prepare_model(detector, gw, parameters)
        samples = m_pymc.run_mcmc(model)
    if method == "emcee":
        model = m_emcee.prepare_model(detector, gw, parameters)
        samples = m_emcee.run_mcmc(model)
    if method == "multinest":
        model = m_multinest.prepare_model(detector, gw, parameters)
        samples = m_multinest.run_mcmc(model)
    
    limit_phi = []
    for i in range(parameters.flux.ncomponents):
        limit_phi.append(upperlimit_from_sample(samples[f"phi{i}"], 0.90))
        logging.getLogger("jang").info("[Limits(MCMC-%s)] %s, %s, %s, limit(Flux%d) = %.3e", method, gw.name, detector.name, parameters.flux, i, limit_phi[-1])
    # limit_eiso = upperlimit_from_sample(samples["eiso"], 0.90)
    # logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(Eiso) = %.3e", gw.name, detector.name, parameters.flux, limit_eiso)
    # limit_etot = upperlimit_from_sample(samples["etot"], 0.90)
    # logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(Etot) = %.3e", gw.name, detector.name, parameters.flux, limit_etot)
    # limit_fnu = upperlimit_from_sample(samples["fnu"], 0.90)
    # logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(fnu) = %.3e", gw.name, detector.name, parameters.flux, limit_fnu)
    
    # return limit_phi, limit_eiso, limit_etot, limit_fnu
    return limit_phi, 0, 0, 0
