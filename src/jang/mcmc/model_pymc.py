import healpy as hp
import logging
import numpy as np
import pymc as pm

from jang.io import NuDetectorBase, GW, Parameters
from jang.io.neutrinos import BackgroundFixed, BackgroundGaussian


def prepare_model(detector: NuDetectorBase, gw: GW, parameters: Parameters) -> pm.model.core.Model:
    
    npix = hp.nside2npix(parameters.nside)
    toysgw = gw.prepare_toys("ra", "dec", "luminosity_distance", "radiated_energy", "theta_jn", nside=parameters.nside)
    toysgw = np.array([[toy.ipix, toy.luminosity_distance, toy.radiated_energy, toy.theta_jn] for toy in toysgw])
    coords = {
        "gw_toys": np.arange(len(toysgw)),
        "gw_pars": ["ipix", "luminosity_distance", "radiated_energy", "theta_jn"],
        "pix": np.arange(npix),
        "nusample": [s.name for s in detector.samples],
        "fluxcomponents": [str(c) for c in parameters.flux.components]
    }
    
    with pm.Model(coords=coords) as model:
        gw = pm.Data("gw", toysgw, dims=("gw_toys", "gw_pars"))
        bkg = []
        for s in detector.samples:
            if not parameters.apply_det_systematics or isinstance(s.background, BackgroundFixed):
                bkg.append(pm.Data(f"bkg_{s.name}", s.background.nominal))
            elif isinstance(s.background, BackgroundGaussian):
                bkg.append(pm.TruncatedNormal(f"bkg_{s.name}", mu=s.background.b0, sigma=s.background.error_b, lower=0))
        if (not parameters.apply_det_systematics) or np.all(detector.error_acceptance == 0):
            xacc = pm.Data("xacc", 1)
        else:
            xacc = pm.MvNormal("xacc", mu=np.ones(detector.nsamples), cov=detector.error_acceptance, dims="nusample")
        itoygw = pm.Categorical("itoygw", p=np.ones(len(toysgw))/len(toysgw))
        parameters.flux.define_signal_parameters()
        parameters.flux.define_auxiliary(gw, itoygw, parameters)
        sig = parameters.flux.define_expected_signal(gw, itoygw, xacc, detector, parameters.nside)
        obs = pm.Poisson("nobs", mu=bkg+sig, observed=[s.nobserved for s in detector.samples], dims="nusample")

    return model


def run_mcmc(model, silent=True):
    
    if silent:
        logger = logging.getLogger("pymc3")
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    with model:
        idata = pm.sample(draws=10000, chains=5, progressbar=not silent)
    samples = {v: idata.posterior[v].values.flatten() for v in ("eiso", "etot", "fnu")}
    for i in range(idata.posterior["phi"].shape[2]):
        samples[f"phi{i}"] = idata.posterior["phi"].values[:,:,i].flatten()
        
    if "xacc" in idata.posterior:
        print(idata.posterior["xacc"].values.flatten())
        
    return samples