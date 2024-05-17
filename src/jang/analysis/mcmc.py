import healpy as hp
import logging
import numpy as np
import pymc as pm
from typing import Dict, Tuple

from jang.io import NuDetectorBase, GW, Parameters
from jang.io.neutrinos import BackgroundFixed, BackgroundGaussian
import jang.utils.conversions


def upperlimit_from_sample(sample: np.ndarray, CL: float = 0.90):
    x = np.array(sample).flatten()
    return np.percentile(x, 100*CL)


def prepare_model(detector: NuDetectorBase, gw: GW, parameters: Parameters) -> pm.model.core.Model:
    
    npix = hp.nside2npix(parameters.nside)
    toysgw = gw.prepare_toys("ra", "dec", "luminosity_distance", "radiated_energy", "theta_jn", nside=parameters.nside)
    toysgw = np.array([[toy.ipix, toy.luminosity_distance, toy.radiated_energy, toy.theta_jn] for toy in toysgw])
    coords = {
        "gw_toys": np.arange(len(toysgw)),
        "gw_pars": ["ipix", "luminosity_distance", "radiated_energy", "theta_jn"],
        "pix": np.arange(npix),
        "nusample": [s.name for s in detector.samples]
    }
    
    with pm.Model(coords=coords) as model:
        gw = pm.Data("gw", toysgw, dims=("gw_toys", "gw_pars"))
        acc_map = pm.Data('acceptance_map', [s.acceptances[parameters.spectrum].map for s in detector.samples], dims=("sample", "pix")) 
        bkg = []
        for s in detector.samples:
            if not parameters.apply_det_systematics or isinstance(s.background, BackgroundFixed):
                bkg.append(pm.Data(f"bkg_{s.name}", s.background.nominal))
            elif isinstance(s.background, BackgroundGaussian):
                bkg.append(pm.TruncatedNormal(f"bkg_{s.name}", mu=s.background.b0, sigma=s.background.error_b, lower=0))
        if not parameters.apply_det_systematics:
            xacc = 1
        else:
            xacc = pm.MvNormal("xacc", mu=np.ones(detector.nsamples), cov=detector.error_acceptance, dims="nusample")
        itoygw = pm.Categorical("itoygw", p=np.ones(len(toysgw))/len(toysgw))
        phi = pm.Uniform("phi", lower=0, upper=1e9)
        ipix = pm.Deterministic("ipix", gw[itoygw][0]).astype("int64")
        eiso = pm.Deterministic("eiso", phi / jang.utils.conversions.eiso_to_phi(parameters.range_energy_integration, parameters.spectrum, gw[itoygw][1]))
        etot = pm.Deterministic("etot", eiso / jang.utils.conversions.etot_to_eiso(gw[itoygw][3], parameters.jet))
        fnu = pm.Deterministic("fnu", etot / jang.utils.conversions.fnu_to_etot(gw[itoygw][2]))
        sig = pm.Deterministic("sig", phi / 6 * xacc * acc_map[:,ipix], dims="sample")
        obs = pm.Poisson("nobs", mu=bkg+sig, observed=[s.nobserved for s in detector.samples], dims="sample")

    return model


def get_samples(model: pm.model.core.Model) -> Dict[str, np.ndarray]:

    with model:
        idata = pm.sample(draws=20000, chains=4)
    return {v: idata.posterior[v].values.flatten() for v in ("phi", "eiso", "etot", "fnu")}
    
    
def get_limits(detector: NuDetectorBase, gw: GW, parameters: Parameters) -> Tuple[float]:
    
    model = prepare_model(detector, gw, parameters)
    samples = get_samples(model)
    
    limit_phi = upperlimit_from_sample(samples["phi"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(Flux) = %.3e", gw.name, detector.name, parameters.spectrum, limit_phi)
    limit_eiso = upperlimit_from_sample(samples["eiso"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(Eiso) = %.3e", gw.name, detector.name, parameters.spectrum, limit_eiso)
    limit_etot = upperlimit_from_sample(samples["etot"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(Etot) = %.3e", gw.name, detector.name, parameters.spectrum, limit_etot)
    limit_fnu = upperlimit_from_sample(samples["fnu"], 0.90)
    logging.getLogger("jang").info("[Limits(MCMC)] %s, %s, %s, limit(fnu) = %.3e", gw.name, detector.name, parameters.spectrum, limit_fnu)
    
    return limit_phi, limit_eiso, limit_etot, limit_fnu
