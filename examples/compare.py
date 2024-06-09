import healpy as hp
import numpy as np
import tempfile
import time

from collections import defaultdict

from jang.io import GW, NuDetector, Parameters
from jang.io.neutrinos import BackgroundGaussian, EffectiveAreaAllSky, EffectiveAreaDeclinationDep
import jang.utils.conversions
import jang.utils.flux as flux
import jang.stats.mcmc as mcmc


tmpdir = tempfile.mkdtemp()

config_str = """
skymap_resolution: 8
detector_systematics: 0

mcmc:
  likelihood: poisson
  priors:
    flux_normalisation: flat
    max_normalisation: 1e10
"""
config_file = f"{tmpdir}/config.yaml"
with open(config_file, "w") as f:
    f.write(config_str)
    
det_str = """
name: TestDet
samples: ["A"]
errors:
  acceptance: 0.10
  acceptance_corr: 0
"""
det_file = f"{tmpdir}/detector.yaml"
with open(det_file, "w") as f:
    f.write(det_str)


def test():
    
    parameters = Parameters(config_file)
    parameters.set_models(flux.FluxVariablePowerLaw(1, 1e6, eref=1e3), jang.utils.conversions.JetIsotropic())
    gw = GW(
            name="GW190412", 
            path_to_fits="examples/input_files/gw_catalogs/GW190412/GW190412_PublicationSamples.fits", 
            path_to_samples="examples/input_files/gw_catalogs/GW190412/GW190412_subset.h5"
    )
    gw.set_parameters(parameters)

    det = NuDetector(det_file)

    class EffAreaTest1(EffectiveAreaAllSky):
        def evaluate(self, energy, ipix, nside):
            return energy**2 * np.exp(-energy/10000)

    det.set_effective_areas([EffAreaTest1()])
    det.set_observations([0], [BackgroundGaussian(0.5, 0.1)])
    
    results = defaultdict(list)
    times = defaultdict(lambda: 0)
    
    N = 1
    parameters.apply_det_systematics = False
    for method in ("emcee", "ultranest", ):
        for _ in range(N):
            t0 = time.time()
            results[f"{method}_nosyst"].append(mcmc.get_limits(det, gw, parameters, method=method)[0][0])
            times[f"{method}_nosyst"] += time.time() - t0

    # N = 20
    # parameters.apply_det_systematics = True
    # for method in ("emcee", "multinest", "ultranest"):
    #     for _ in range(N):
    #         t0 = time.time()
    #         results[f"{method}_wsyst"].append(mcmc.get_limits(det, gw, parameters, method=method)[0][0])
    #         times[f"{method}_wsyst"] += time.time() - t0
            
    # compute naive upper limits
    nside = 8
    best_ipix = np.argmax(gw.fits.get_skymap(nside))
    parameters.flux.set_shapes([2])
    acc = EffAreaTest1().compute_acceptance(parameters.flux.components[0], best_ipix, nside)
    print(f"Naive UL = {2.3 / (acc/6):.2e}")

    for k in results.keys():
        print(f"{k:25s} => {np.average(results[k]):.2e} ± {np.std(results[k]):.2e}, TIME = {times[k]/N:.2f} s")


if __name__ == "__main__":
    
    test()