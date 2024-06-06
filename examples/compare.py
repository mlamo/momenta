import healpy as hp
import numpy as np
import tempfile
import time

from collections import defaultdict

from jang.io import GW, NuDetector, Parameters
from jang.io.neutrinos import BackgroundGaussian, EffectiveAreaDeclinationDep
import jang.utils.conversions
import jang.analysis.limits as limits
import jang.mcmc.mcmc as mcmc
import jang.utils.flux as flux


tmpdir = tempfile.mkdtemp()

config_str = """
analysis:
  nside: 8
  apply_det_systematics: 0
  ntoys_det_systematics: 1000
  search_region: fullsky
  likelihood: poisson
  prior_signal: flat

range:
  log10_flux: [-5, 5, 1000]
  log10_etot: [48, 62, 1400]
  log10_fnu: [-5, 10, 1500]
  neutrino_energy_GeV: [1, 1e6]
"""
config_file = f"{tmpdir}/config.yaml"
with open(config_file, "w") as f:
    f.write(config_str)
    
det_str = """
name: TestDet

nsamples: 1
samples:
  names: ["A"]
  shortnames: ["A"]
  energyrange: [1, 1e6]

errors:
  acceptance: 0.10
  acceptance_corr: 0
"""
det_file = f"{tmpdir}/detector.yaml"
with open(det_file, "w") as f:
    f.write(det_str)


def test():
    
    parameters = Parameters(config_file)
    parameters.set_models("x**-2", jang.utils.conversions.JetIsotropic())
    parameters.nside = 8
    gw = GW(
            "GW190412", 
            "examples/input_files/gw_catalogs/GW190412/GW190412_PublicationSamples.fits", 
            "examples/input_files/gw_catalogs/GW190412/GW190412_subset.h5"
    )
    gw.set_parameters(parameters)

    det = NuDetector(det_file)

    class EffAreaTest1(EffectiveAreaDeclinationDep):
        def __init__(self):
            super().__init__()
            self.func = lambda energy, dec: (dec+90)/180 * energy**2 * np.exp(-energy/10000)

    class EffAreaTest2(EffectiveAreaDeclinationDep):
        def __init__(self):
            super().__init__()
            self.func = lambda energy, dec: (dec+90)/180 * energy**2 * np.exp(-energy/10000)

    parameters.flux = flux.FluxFixedPowerLaw(1, 1000000, 2)
    det.set_effective_areas([EffAreaTest1()])
    det.set_observations([0], [BackgroundGaussian(0.5, 0.1)])
    
    results = defaultdict(list)
    times = defaultdict(lambda: 0)
    
    N = 1
    parameters.apply_det_systematics = False
    for _ in range(N):
        t0 = time.time()
        results["old_nosyst"].append(limits.get_limit_flux(det, gw, parameters))
        times["old_nosyst"] += time.time() - t0
        t0 = time.time()
        results["pymc_nosyst"].append(mcmc.get_limits(det, gw, parameters, method="pymc")[0][0])
        times["pymc_nosyst"] += time.time() - t0
    #     t0 = time.time()
    #     results["emcee_nosyst"].append(mcmc.get_limits(det, gw, parameters, method="emcee")[0][0])
    #     times["emcee_nosyst"] += time.time() - t0
    #     t0 = time.time()
    #     results["multinest_nosyst"].append(mcmc.get_limits(det, gw, parameters, method="multinest")[0][0])
    #     times["multinest_nosyst"] += time.time() - t0

    N = 1
    parameters.apply_det_systematics = True
    for _ in range(N):
        t0 = time.time()
        results["old_wsyst"].append(limits.get_limit_flux(det, gw, parameters))
        times["old_wsyst"] += time.time() - t0
        t0 = time.time()
        results["pymc_wsyst"].append(mcmc.get_limits(det, gw, parameters, method="pymc")[0][0])
        times["pymc_wsyst"] += time.time() - t0
        # t0 = time.time()
        # results["emcee_wsyst"].append(mcmc.get_limits(det, gw, parameters, method="emcee")[0][0])
        # times["emcee_wsyst"] += time.time() - t0
        # t0 = time.time()
        # results["multinest_wsyst"].append(mcmc.get_limits(det, gw, parameters, method="multinest")[0][0])
        # times["multinest_wsyst"] += time.time() - t0

    # compute naive upper limits
    nside = 8
    best_ipix = np.argmax(gw.fits.get_skymap(nside))
    acc = EffAreaTest1().compute_acceptance(parameters.flux.components[0], best_ipix, nside)
    print(f"Naive UL = {2.3 / (acc/6):.2e}")

    for k in results.keys():
        print(f"{k:25s} => {np.average(results[k]):.2e} ± {np.std(results[k]):.2e}, TIME = {times[k]/N:.2f} s")


if __name__ == "__main__":
    
    test()