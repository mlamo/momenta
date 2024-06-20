import numpy as np
import tempfile
import time

from collections import defaultdict

from jang.io import GW, NuDetector, Parameters
from jang.io.neutrinos import BackgroundGaussian, EffectiveAreaAllSky
from jang.stats.run import run
from jang.stats.limits import get_limits, get_limits_with_uncertainties
import jang.utils.conversions
import jang.utils.flux as flux


tmpdir = tempfile.mkdtemp()

config_str = """
skymap_resolution: 8
detector_systematics: 0

analysis:
  likelihood: poisson
  prior_normalisation:
    type: flat-linear
    range: [1.0e-10, 1.0e+10]
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
    parameters.set_models(flux.FluxFixedPowerLaw(1, 1e6, 2, eref=1), jang.utils.conversions.JetIsotropic())
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
    uncertainties = defaultdict(list)
    times = defaultdict(lambda: 0)
    
    N = 20
    parameters.apply_det_systematics = False
    parameters.prior_normalisation = "flat-linear"
    for method in ("emcee", "multinest", "ultranest", ):
        for _ in range(N):
            t0 = time.time()
            model, result = run(det, gw, parameters, method)
            if "weighted_samples" in result:
                limits = get_limits_with_uncertainties(result["weighted_samples"], model)["flux0_norm"]
                limit, unc = limits[0], limits[1]
            else:
                limit = get_limits(result["samples"], model)["flux0_norm"]
                unc = np.nan
            results[f"{method}_flatlin"].append(limit)
            uncertainties[f"{method}_flatlin"].append(unc)
            times[f"{method}_flatlin"] += time.time() - t0

    N = 20
    parameters.apply_det_systematics = False
    parameters.prior_normalisation = "flat-log"
    for method in ("emcee", "multinest", "ultranest", ):
        for _ in range(N):
            t0 = time.time()
            model, result = run(det, gw, parameters, method)
            if "weighted_samples" in result:
                limits = get_limits_with_uncertainties(result["weighted_samples"], model)["flux0_norm"]
                limit, unc = limits[0], limits[1]
            else:
                limit = get_limits(result["samples"], model)["flux0_norm"]
                unc = np.nan
            results[f"{method}_flatlog"].append(limit)
            uncertainties[f"{method}_flatlog"].append(unc)
            times[f"{method}_flatlog"] += time.time() - t0
            
    N = 20
    parameters.apply_det_systematics = False
    parameters.prior_normalisation = "jeffreys-pois"
    for method in ("emcee", "multinest", "ultranest", ):
        for _ in range(N):
            t0 = time.time()
            model, result = run(det, gw, parameters, method)
            if "weighted_samples" in result:
                limits = get_limits_with_uncertainties(result["weighted_samples"], model)["flux0_norm"]
                limit, unc = limits[0], limits[1]
            else:
                limit = get_limits(result["samples"], model)["flux0_norm"]
                unc = np.nan
            results[f"{method}_jeffreyspois"].append(limit)
            uncertainties[f"{method}_jeffreyspois"].append(unc)
            times[f"{method}_jeffreyspois"] += time.time() - t0
            
    # compute naive upper limits
    nside = 8
    best_ipix = np.argmax(gw.fits.get_skymap(nside))
    parameters.flux.set_shapes([2])
    acc = EffAreaTest1().compute_acceptance(parameters.flux.components[0], best_ipix, nside)
    print(f"Naive UL = {2.3 / (acc/6):.2e}")

    for k in results.keys():
        print(f"{k:25s} => {np.average(results[k]):.2e} ± {np.std(results[k]):.2e} ({np.average(uncertainties[k]):.2e}), TIME = {times[k]/N:.2f} s")


if __name__ == "__main__":
    
    test()